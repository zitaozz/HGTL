import numpy as np
import torch
from collections import defaultdict
import torch.nn.functional as F
from tqdm import tqdm


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class HGTL(torch.nn.Module):
    def __init__(self, user_num, item_num, category, category_emb, category_num, User_Item, Item_User, category_contain, args, max_category_len = 6, max_neighbor_len = 10):
        super(HGTL, self).__init__()

        self.user_num = user_num + 1
        self.item_num = item_num + 1
        self.category = category
        self.category_num = category_num
        self.max_category_len = max_category_len
        self.dev = args.device
        self.User_Item = User_Item
        self.Item_User = Item_User
        self.category_contain = category_contain
        self.max_neighbor_len = max_neighbor_len

        self.item_emb = torch.nn.Embedding(self.item_num, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num, args.hidden_units, padding_idx=0)
        self.category_emb = self.get_category_emb(category_emb, args)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self._init_layer_lists(args)
        self._init_last_layernorms(args)
        self._init_user_cross_layers(args)

        self.UIU_adj = self.get_UIU_adj()
        self.UICIU_adj = self.get_UICIU_adj()
        self.CIC_adj = self.get_CIC_adj()
        self.CIUIC_adj = self.get_CIUIC_adj()
        self.IUI_adj = self.get_IUI_adj()
        self.ICI_adj = self.get_ICI_adj()

    def _init_layer_lists(self, args):
        layer_types = [
            ('attention',),
            ('cross_attention',),
            ('cross_attention3',),
            ('forward',),
            ('cross_forward',),
            ('cross_forward3',),
            ('ca_attention',),
            ('ca_forward',)
        ]
        for layer_type in layer_types:
            layernorm_list = getattr(self, f"{layer_type[0]}_layernorms", torch.nn.ModuleList())
            layer_list = getattr(self, f"{layer_type[0]}_layers", torch.nn.ModuleList())
            for _ in range(args.num_blocks):
                layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
                layernorm_list.append(layernorm)
                if 'attention' in layer_type[0]:
                    layer = torch.nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate)
                else:
                    layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
                layer_list.append(layer)
            setattr(self, f"{layer_type[0]}_layernorms", layernorm_list)
            setattr(self, f"{layer_type[0]}_layers", layer_list)

        self.ic_attention_layernorms = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.ic_attention_layers = torch.nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate)
        self.ic_forward_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.ic_forward_layers = PointWiseFeedForward(args.hidden_units, args.dropout_rate)

    def _init_last_layernorms(self, args):
        last_layernorm_names = [
            'last_layernorm', 'last_cross_layernorm', 'last_cross_layernorm3',
            'ca_last_layernorm', 'ic_last_layernorm', 'lc_last_layernorm',
            'lc_last_layernorm2', 'lc_last_layernorm3', 'iui_last_layernorm',
            'iui_attention_layernorms', 'cic_last_layernorm', 'cic_attention_layernorms',
            'uiu_last_layernorm', 'uiu_attention_layernorms', 'uiciu_last_layernorm',
            'uiciu_attention_layernorms', 'ciuic_last_layernorm', 'ciuic_attention_layernorms',
            'ici_last_layernorm', 'ici_attention_layernorms'
        ]
        for name in last_layernorm_names:
            setattr(self, name, torch.nn.LayerNorm(args.hidden_units, eps=1e-8))

    def _init_user_cross_layers(self, args):
        for i in range(1, 7):
            setattr(self, f"last_user_cross_layernorm{i}", torch.nn.LayerNorm(4 * args.hidden_units if i < 4 else args.hidden_units, eps=1e-8))
            setattr(self, f"dropout{i}", torch.nn.Dropout(p=args.dropout_rate))
            setattr(self, f"gating{i}", torch.nn.Linear(4 * args.hidden_units, args.hidden_units))

    def get_category_emb(self, category_emb, args):
        embedding_layer = torch.nn.Embedding(self.category_num, args.hidden_units, padding_idx=0)
        linear_layer = torch.nn.Linear(768, args.hidden_units)
        category_emb = linear_layer(category_emb)
        embedding_layer.weight = torch.nn.Parameter(category_emb)
        return embedding_layer

    def get_category_adj(self):
        A = np.zeros((self.category_num, self.category_num))
        N = np.zeros(self.category_num)
        M = np.zeros((self.category_num, self.category_num))

        for values in self.category.values():
            for idx, v in enumerate(values):
                N[v] += 1
                for j in range(idx + 1, len(values)):
                    if values[j] != 0:
                        M[v][values[j]] += 1

        for i in range(self.category_num):
            for j in range(self.category_num):
                A[i][j] = M[i][j] / N[i]

        indices = []
        for i, row in enumerate(A):
            row_indices = []
            for j, val in enumerate(row):
                if val > 0.5:
                    row_indices.append(j)
                    indices.append([i, j])
            if len(row_indices) > 0:
                print(f"Row {i}: {row_indices}")

        A = norm(A)
        return torch.Tensor(A)

    def get_UIU_adj(self):
        Adj = {}
        k = self.max_neighbor_len
        for user in tqdm(range(1, self.user_num), desc="get_UIU_adj", ncols=80):
            user_count = {}
            items = self.User_Item[user]
            for item in items:
                neighbor_users = self.Item_User[item]
                for neighbor_user in neighbor_users:
                    if neighbor_user in user_count:
                        user_count[neighbor_user] += 1
                    else:
                        user_count[neighbor_user] = 1
            sorted_users = sorted(user_count.items(), key=lambda x: x[1], reverse=True)
            Adj[user] = [sorted_users[i][0] for i in range(min(k, len(sorted_users)))]
        return Adj

    def get_UICIU_adj(self):
        Adj = {}
        k = self.max_neighbor_len
        for user in tqdm(range(1, self.user_num), desc="get_UICIU_adj", ncols=80):
            user_count = {}
            items = self.User_Item[user]
            category_count = {}
            for item in items:
                # neighbor_users = self.Item_User[item]
                categories = self.category[item]
                if categories[0] == 0:
                    continue
                for c in categories:
                    if c in category_count:
                        category_count[c] += 1
                    else:
                        category_count[c] = 1
            sorted_categoies = sorted(category_count.items(), key=lambda x: x[1], reverse=True)
            top_categories = [sorted_categoies[i][0] for i in range(min(k, len(sorted_categoies)))]
            for c in top_categories:
                categories_items = self.category_contain[c]
                for categories_item in categories_items:
                    neighbor_users = self.Item_User[categories_item]
                    for neighbor_user in neighbor_users:
                        if neighbor_user in user_count:
                            user_count[neighbor_user] += 1
                        else:
                            user_count[neighbor_user] = 1
            sorted_users = sorted(user_count.items(), key=lambda x: x[1], reverse=True)
            Adj[user] = [sorted_users[i][0] for i in range(min(k, len(sorted_users)))]
        return Adj

    def get_CIC_adj(self):
        Adj = {}
        k = self.max_neighbor_len
        for c in tqdm(range(1, self.category_num), desc="get_CIC_adj", ncols=80):
            category_count = {}
            items = self.category_contain[c]
            for item in items:
                neighbor_categories = self.category[item]
                if neighbor_categories[0] == 0:
                    continue
                for neighbor_category in neighbor_categories:
                    if neighbor_category in category_count:
                        category_count[neighbor_category] += 1
                    else:
                        category_count[neighbor_category] = 1
            sorted_categories = sorted(category_count.items(), key=lambda x: x[1], reverse=True)
            Adj[c] = [sorted_categories[i][0] for i in range(min(k, len(sorted_categories)))]
        return Adj

    def get_CIUIC_adj(self):
        Adj = {}
        k = self.max_neighbor_len
        for c in tqdm(range(1, self.category_num), desc="get_CIUIC_adj", ncols=80):
            category_count = {}
            items = self.category_contain[c]
            user_count = {}
            for item in items:
                users = self.Item_User[item]
                for u in users:
                    if u in user_count:
                        user_count[u] += 1
                    else:
                        user_count[u] = 1
            sorted_users = sorted(user_count.items(), key=lambda x: x[1], reverse=True)
            top_users = [sorted_users[i][0] for i in range(min(k, len(sorted_users)))]
            for u in top_users:
                users_items = self.User_Item[u]
                for user_item in users_items:
                    neighbor_categories = self.category[user_item]
                    for neighbor_category in neighbor_categories:
                        if neighbor_category in category_count:
                            category_count[neighbor_category] += 1
                        else:
                            category_count[neighbor_category] = 1
            sorted_categories = sorted(category_count.items(), key=lambda x: x[1], reverse=True)
            Adj[c] = [sorted_categories[i][0] for i in range(min(k, len(sorted_categories)))]
        return Adj

    def get_IUI_adj(self):
        Adj = {}
        k = self.max_neighbor_len
        for item in tqdm(range(1, self.item_num), desc="get_IUI_adj", ncols=80):
            item_count = {}
            users = self.Item_User[item]
            for user in users:
                user_items = self.User_Item[user]
                for neighbor_item in user_items:
                    if neighbor_item in item_count:
                        item_count[neighbor_item] += 1
                    else:
                        item_count[neighbor_item] = 1

            sorted_items = sorted(item_count.items(), key=lambda x: x[1], reverse=True)
            Adj[item] = [sorted_items[i][0] for i in range(min(k, len(sorted_items)))]
        return Adj

    def get_ICI_adj(self):
        k = self.max_neighbor_len
        item_num = self.item_num
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        indices = []
        values = []
        for category_id, items in self.category_contain.items():
            for item in items:
                indices.append([category_id, item])
                values.append(1)
        indices = torch.tensor(indices, dtype=torch.long).t().to(device)
        values = torch.tensor(values, dtype=torch.float32).to(device)
        category_contain_sparse = torch.sparse_coo_tensor(indices, values,
                                                          (max(self.category_contain.keys()) + 1, item_num),
                                                          device=device)
        Adj = {}

        for item in tqdm(range(1, item_num), desc="get_ICI_adj", ncols=80):
            item_categories = self.category[item]
            if item_categories[0] == 0:
                Adj[item] = [0] * k
                continue

            item_count = torch.zeros(item_num, dtype=torch.float32, device=device)

            for category_id in item_categories:
                if category_id >= category_contain_sparse.size(0):
                    continue
                neighbor_items = category_contain_sparse[category_id].to_dense()
                item_count += neighbor_items

            sorted_items = torch.argsort(item_count, descending=True)
            top_k = sorted_items[:k].cpu().tolist()
            Adj[item] = top_k

        return Adj

    def get_item_embedding(self, log_seqs):
        seqsi1 = self.get_item_metapath_embedding_IUI(log_seqs)
        seqsi2 = self.get_item_metapath_embedding_ICI(log_seqs)
        seqs = seqsi1 + seqsi2
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)
        return seqs, timeline_mask

    def apply_attention_block(self, query_seq, key_value_seq, attention_layers, attention_layernorms,
                              forward_layers, forward_layernorms, attention_mask, timeline_mask, att_key=False):
        for i in range(len(attention_layers)):
            query_seq = torch.transpose(query_seq, 0, 1)
            if att_key:
                key_value_seq_transposed = torch.transpose(key_value_seq, 0, 1)
            else:
                key_value_seq_transposed = query_seq

            Q = attention_layernorms[i](query_seq)
            mha_outputs, _ = attention_layers[i](Q, key_value_seq_transposed, key_value_seq_transposed,
                                                 attn_mask=attention_mask)
            query_seq = Q + mha_outputs
            query_seq = torch.transpose(query_seq, 0, 1)
            if not att_key:
                key_value_seq = query_seq

            query_seq = forward_layernorms[i](query_seq)
            query_seq = forward_layers[i](query_seq)
            query_seq *= ~timeline_mask.unsqueeze(-1)
        return query_seq

    def get_attention_mask(self, log_seqs, mask=None):
        if mask is None:
            tl = log_seqs.shape[1]
            return ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        else:
            batch_size, tl = log_seqs.shape
            attention_mask = np.ones((batch_size, tl, tl), dtype=bool)
            for b in range(batch_size):
                for i in range(tl):
                    attention_mask[b][i][0:mask[b][i]] = False
            attention_mask[:, :, 0] = False
            return torch.from_numpy(attention_mask).to(self.dev)

    def get_category_loss_and_features(self, log_seqs_list):
        c1 = self.get_category_metapath_embedding_CIC()
        c2 = self.get_category_metapath_embedding_CIUIC()
        con_loss2 = SSL_binary(c1, c2)
        self.category_emb.weight = torch.nn.Parameter(c1 + c2)
        category_features_list = []
        att_category_features_list = []
        for log_seqs in log_seqs_list:
            category_features = self.get_item_category_att(log_seqs)
            att_category_features = self.category_attention(log_seqs, category_features)
            category_features_list.append(category_features)
            att_category_features_list.append(att_category_features)
        return con_loss2, category_features_list, att_category_features_list

    def get_user_embedding(self, user_ids):
        u1 = self.get_user_metapath_embedding_UIU(user_ids)
        u2 = self.get_user_metapath_embedding_UICIU(user_ids)
        con_loss3 = SSL_binary(u1, u2)
        u = u1 + u2
        u *= self.user_emb.embedding_dim ** 0.5
        u = self.emb_dropout(u)
        return u, con_loss3

    def process_single_seq(self, log_seqs, attention_layers, attention_layernorms,
                           forward_layers, forward_layernorms, mask=None, att_key=False):
        seqs, timeline_mask = self.get_item_embedding(log_seqs)
        attention_mask = self.get_attention_mask(log_seqs, mask)
        return self.apply_attention_block(seqs, seqs, attention_layers, attention_layernorms,
                                          forward_layers, forward_layernorms, attention_mask, timeline_mask, att_key)

    def get_contrastive_loss(self, seqsi1, seqsi2):
        return SSL(seqsi1, seqsi2)

    def log2feats(self, user_ids, log_seqs, log_seqs2, log_seqs3, mask1=None, mask2=None, mask3=None, target=None):
        attention_mapping = {
            "A": [False, True, True],
            "B": [True, False, True],
            "C": [True, True, False]
        }

        seqs_info = [
            self.process_single_seq(log_seqs, self.attention_layers, self.attention_layernorms,
                                    self.forward_layers, self.forward_layernorms, mask1, att_key=attention_mapping[target][0]),
            self.process_single_seq(log_seqs2, self.cross_attention_layers, self.cross_attention_layernorms,
                                    self.cross_forward_layers, self.cross_forward_layernorms, mask2, att_key=attention_mapping[target][1]),
            self.process_single_seq(log_seqs3, self.cross_attention3_layers, self.cross_attention3_layernorms,
                                    self.cross_forward3_layers, self.cross_forward3_layernorms, mask3, att_key=attention_mapping[target][2])
        ]
        att_seq, att_seq2, att_seq3 = [info for info in seqs_info]

        seqsi_list = [
            (self.get_item_metapath_embedding_IUI(log_seqs), self.get_item_metapath_embedding_ICI(log_seqs)),
            (self.get_item_metapath_embedding_IUI(log_seqs2), self.get_item_metapath_embedding_ICI(log_seqs2)),
            (self.get_item_metapath_embedding_IUI(log_seqs3), self.get_item_metapath_embedding_ICI(log_seqs3))
        ]
        target_index_mapping = {
            "A": 0,
            "B": 1,
            "C": 2
        }
        con_loss1 = self.get_contrastive_loss(*seqsi_list[target_index_mapping[target]])

        con_loss2, category_features_list, att_category_features_list = self.get_category_loss_and_features(
            [log_seqs, log_seqs2, log_seqs3])

        u, con_loss3 = self.get_user_embedding(user_ids)
        u = u.unsqueeze(1)
        u = u.expand_as(att_seq)

        layer_mapping = {
            "A": (self.last_user_cross_layernorm4, self.dropout4, self.gating4, self.last_user_cross_layernorm1, self.dropout1, self.gating1, self.lc_last_layernorm),
            "B": (self.last_user_cross_layernorm5, self.dropout5, self.gating5, self.last_user_cross_layernorm2, self.dropout2, self.gating2, self.lc_last_layernorm2),
            "C": (self.last_user_cross_layernorm6, self.dropout6, self.gating6, self.last_user_cross_layernorm3, self.dropout3, self.gating3, self.lc_last_layernorm3)
        }
        cross_layernorm, dropout_cross, gating_cross, user_layernorm, dropout_user, gating_user, lc_layernorm = layer_mapping[target]

        log_feats = self.last_layernorm(att_seq)
        log_feats2 = self.last_cross_layernorm(att_seq2)
        log_feats3 = self.last_cross_layernorm3(att_seq3)

        cross_input_mapping = {
            "A": (log_feats2, log_feats3, att_category_features_list[1], att_category_features_list[2]),
            "B": (log_feats, log_feats3, att_category_features_list[0], att_category_features_list[2]),
            "C": (log_feats, log_feats2, att_category_features_list[0], att_category_features_list[1])
        }
        cross_input = torch.cat(cross_input_mapping[target], dim=2)
        cross = cross_layernorm(dropout_cross(gating_cross(cross_input)))

        user_input_mapping = {
            "A": (log_feats, att_category_features_list[0], cross, u),
            "B": (log_feats2, att_category_features_list[1], cross, u),
            "C": (log_feats3, att_category_features_list[2], cross, u)
        }
        user = torch.cat(user_input_mapping[target], dim=2)
        user = user_layernorm(user)
        user = dropout_user(gating_user(user))
        user = lc_layernorm(user)

        return user, con_loss1, con_loss2, con_loss3

    def get_user_metapath_embedding_UIU(self, user_ids):
        users = self.user_emb(torch.LongTensor(user_ids).to(self.dev))
        users *= self.user_emb.embedding_dim ** 0.5
        users = self.emb_dropout(users)

        neighbor_users = []
        for u in user_ids:
            neighbor = self.UIU_adj[u]
            while len(neighbor) < self.max_neighbor_len:
                neighbor.append(0)
            neighbor_users.append(neighbor)
        neighbor_users = torch.LongTensor(neighbor_users)

        Q = self.uiu_attention_layernorms(users)
        Q = Q.unsqueeze(1)

        K = self.user_emb(neighbor_users.to(self.dev))
        K *= self.user_emb.embedding_dim ** 0.5
        K = V = self.emb_dropout(K)
        dot_products = torch.sum(Q * K, dim=-1)
        scaled_dot_products = dot_products / 50
        attention_weights = torch.softmax(scaled_dot_products, dim=-1)
        output = torch.sum(attention_weights.unsqueeze(-1) * V, dim=1)

        user_features = self.uiu_last_layernorm(output)
        return user_features

    def get_user_metapath_embedding_UICIU(self, user_ids):
        users = self.user_emb(torch.LongTensor(user_ids).to(self.dev))
        users *= self.user_emb.embedding_dim ** 0.5
        users = self.emb_dropout(users)

        neighbor_users = []
        for u in user_ids:
            neighbor = self.UICIU_adj[u]
            while len(neighbor) < self.max_neighbor_len:
                neighbor.append(0)
            neighbor_users.append(neighbor)
        neighbor_users = torch.LongTensor(neighbor_users)

        Q = self.uiciu_attention_layernorms(users)
        Q = Q.unsqueeze(1)

        K = self.user_emb(neighbor_users.to(self.dev))
        K *= self.user_emb.embedding_dim ** 0.5
        K = V = self.emb_dropout(K)
        dot_products = torch.sum(Q * K, dim=-1)
        scaled_dot_products = dot_products / 50
        attention_weights = torch.softmax(scaled_dot_products, dim=-1)
        output = torch.sum(attention_weights.unsqueeze(-1) * V, dim=1)

        user_features = self.uiciu_last_layernorm(output)
        return user_features

    def get_category_metapath_embedding_CIC(self):
        category_ids = list(range(0, self.category_num))
        categories = self.category_emb(torch.LongTensor(category_ids).reshape(-1, 1).to(self.dev))
        categories *= self.category_emb.embedding_dim ** 0.5
        categories = self.emb_dropout(categories)

        neighbor_categories = []
        for c in category_ids:
            if c == 0:
                neighbor_categories.append([0] * self.max_neighbor_len)
                continue
            neighbor = self.CIC_adj[c]
            while len(neighbor) < self.max_neighbor_len:
                neighbor.append(0)
            neighbor_categories.append(neighbor)
        neighbor_categories = torch.LongTensor(neighbor_categories)

        Q = self.cic_attention_layernorms(categories)

        K = self.category_emb(neighbor_categories.to(self.dev))
        K *= self.category_emb.embedding_dim ** 0.5
        K = V = self.emb_dropout(K)
        dot_products = torch.sum(Q * K, dim=-1)
        scaled_dot_products = dot_products / 50
        attention_weights = torch.softmax(scaled_dot_products, dim=-1)
        output = torch.sum(attention_weights.unsqueeze(-1) * V, dim=1)

        category_features = self.cic_last_layernorm(output)
        return category_features

    def get_category_metapath_embedding_CIUIC(self):
        category_ids = list(range(0, self.category_num))
        categories = self.category_emb(torch.LongTensor(category_ids).reshape(-1, 1).to(self.dev))
        categories *= self.category_emb.embedding_dim ** 0.5
        categories = self.emb_dropout(categories)

        neighbor_categories = []
        for c in category_ids:
            if c == 0:
                neighbor_categories.append([0] * self.max_neighbor_len)
                continue
            neighbor = self.CIUIC_adj[c]
            while len(neighbor) < self.max_neighbor_len:
                neighbor.append(0)
            neighbor_categories.append(neighbor)
        neighbor_categories = torch.LongTensor(neighbor_categories)

        Q = self.ciuic_attention_layernorms(categories)
        K = self.category_emb(neighbor_categories.to(self.dev))
        K *= self.category_emb.embedding_dim ** 0.5
        K = V = self.emb_dropout(K)
        dot_products = torch.sum(Q * K, dim=-1)
        scaled_dot_products = dot_products / 50
        attention_weights = torch.softmax(scaled_dot_products, dim=-1)
        output = torch.sum(attention_weights.unsqueeze(-1) * V, dim=1)

        category_features = self.ciuic_last_layernorm(output)
        return category_features

    def get_item_metapath_embedding_IUI(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)

        neighbor_seqs = []
        for b in log_seqs:
            neighbor_seq = []
            for j in b:
                if j == 0:
                    neighbor_seq.append([0] * self.max_neighbor_len)
                else:
                    cd = self.IUI_adj[j]
                    while len(cd) < self.max_neighbor_len:
                        cd.append(0)
                    neighbor_seq.append(cd)
            neighbor_seqs.append(neighbor_seq)
        neighbor_seqs = torch.LongTensor(neighbor_seqs)

        Q = self.iui_attention_layernorms(seqs)
        Q = Q.unsqueeze(2)

        K = self.item_emb(neighbor_seqs.to(self.dev))
        K *= self.item_emb.embedding_dim ** 0.5
        K = V = self.emb_dropout(K)
        dot_products = torch.sum(Q * K, dim=-1)
        scaled_dot_products = dot_products / 50
        attention_weights = torch.softmax(scaled_dot_products, dim=-1)
        output = torch.sum(attention_weights.unsqueeze(-1) * V, dim=2)

        item_features = self.iui_last_layernorm(output)
        return item_features

    def get_item_metapath_embedding_ICI(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)

        neighbor_seqs = []
        for b in log_seqs:
            neighbor_seq = []
            for j in b:
                if j == 0:
                    neighbor_seq.append([0] * self.max_neighbor_len)
                else:
                    cd = self.ICI_adj[j]
                    while len(cd) < self.max_neighbor_len:
                        cd.append(0)
                    neighbor_seq.append(cd)
            neighbor_seqs.append(neighbor_seq)
        neighbor_seqs = torch.LongTensor(neighbor_seqs)

        Q = self.ici_attention_layernorms(seqs)
        Q = Q.unsqueeze(2)

        K = self.item_emb(neighbor_seqs.to(self.dev))
        K *= self.item_emb.embedding_dim ** 0.5
        K = V = self.emb_dropout(K)
        dot_products = torch.sum(Q * K, dim=-1)
        scaled_dot_products = dot_products / 50
        attention_weights = torch.softmax(scaled_dot_products, dim=-1)
        output = torch.sum(attention_weights.unsqueeze(-1) * V, dim=2)

        item_features = self.ici_last_layernorm(output)
        return item_features

    def get_item_category_att(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)

        ca_seqs = []
        for b in log_seqs:
            ca_seq = []
            for j in b:
                cd = self.category[j]
                if len(cd) > self.max_category_len:
                    cd = cd[:self.max_category_len]

                while len(cd) < self.max_category_len:
                    cd.append(0)
                ca_seq.append(cd)
            ca_seqs.append(ca_seq)
        ca_seqs = torch.LongTensor(ca_seqs)

        Q = self.ic_attention_layernorms(seqs)
        Q = Q.unsqueeze(2)

        K = self.category_emb(ca_seqs.to(self.dev))
        K *= self.category_emb.embedding_dim ** 0.5
        K = V = self.emb_dropout(K)

        dot_products = torch.sum(Q * K, dim=-1)
        scaled_dot_products = dot_products / 50
        attention_weights = torch.softmax(scaled_dot_products, dim=-1)
        output = torch.sum(attention_weights.unsqueeze(-1) * V, dim=2)

        category_features = self.ic_last_layernorm(output)
        return category_features

    def category_attention(self, log_seqs, category_features):
        category_features *= self.category_emb.embedding_dim ** 0.5
        category_features = self.emb_dropout(category_features)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        category_features *= ~timeline_mask.unsqueeze(-1)

        tl = category_features.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.ca_attention_layers)):
            category_features = torch.transpose(category_features, 0, 1)
            Q = self.ca_attention_layernorms[i](category_features)
            mha_outputs, _ = self.ca_attention_layers[i](Q, category_features, category_features,
                                                         attn_mask=attention_mask)
            category_features = Q + mha_outputs
            category_features = torch.transpose(category_features, 0, 1)

            category_features = self.ca_forward_layernorms[i](category_features)
            category_features = self.ca_forward_layers[i](category_features)
            category_features *= ~timeline_mask.unsqueeze(-1)

        category_features = self.ca_last_layernorm(category_features)
        return category_features

    def forward(self, user_ids, log_seqs, log_seqs2, log_seqs3, pos_seqs, neg_seqs, mask1, mask2, target):
        if target == "A":
            log_feats, con_loss1, con_loss2, con_loss3 = self.log2feats(user_ids, log_seqs, log_seqs2, log_seqs3, None, mask1,
                                                                        mask2, target)
        elif target == "B":
            log_feats, con_loss1, con_loss2, con_loss3 = self.log2feats(user_ids, log_seqs, log_seqs2, log_seqs3,
                                                                         mask1, None,
                                                                         mask2, target)
        else:
            log_feats, con_loss1, con_loss2, con_loss3 = self.log2feats(user_ids, log_seqs, log_seqs2, log_seqs3,
                                                                         mask1,
                                                                         mask2, None, target)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)


        return pos_logits, neg_logits, con_loss1 / (
                    log_feats.shape[0] * log_feats.shape[1]), con_loss2 / self.category_num, con_loss3 / \
               log_feats.shape[0]

    def predict(self, user_ids, log_seqs, log_seqs2, log_seqs3, item_indices, mask1, mask2, target):
        if target == "A":
            log_feats, con_loss1, con_loss2, con_loss3 = self.log2feats(user_ids, log_seqs, log_seqs2, log_seqs3, None, mask1,
                                                                        mask2, "A")
        elif target == "B":
            log_feats, con_loss1, con_loss2, con_loss3 = self.log2feats(user_ids, log_seqs, log_seqs2, log_seqs3,
                                                                         mask1, None,
                                                                         mask2, "B")
        else:
            log_feats, con_loss1, con_loss2, con_loss3 = self.log2feats(user_ids, log_seqs, log_seqs2, log_seqs3,
                                                                         mask1,
                                                                         mask2, None, "C")

        final_feat = log_feats[:, -1, :]

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits

def norm(adj):
    adj += np.eye(adj.shape[0])
    degree = np.array(adj.sum(1))
    degree = np.diag(np.power(degree, -0.5))
    return degree.dot(adj).dot(degree)

def SSL(sess_emb_hgnn, sess_emb_lgcn):
    def row_column_shuffle(embedding):
        corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
        corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
        return corrupted_embedding

    def score(x1, x2):
        return torch.sum(torch.mul(x1, x2), 2)

    pos = score(sess_emb_hgnn, sess_emb_lgcn)
    neg1 = score(sess_emb_hgnn, row_column_shuffle(sess_emb_lgcn))
    one = torch.ones((neg1.shape[0], neg1.shape[1])).cuda()
    con_loss = torch.sum(-torch.log(1e-7 + torch.sigmoid(pos)) - torch.log(1e-7 + (one - torch.sigmoid(neg1))))
    return con_loss

def SSL_binary(sess_emb_hgnn, sess_emb_lgcn):
    def row_column_shuffle(embedding):
        corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
        return corrupted_embedding

    def score(x1, x2):
        return torch.sum(torch.mul(x1.detach(), x2.detach()), 1)

    pos = score(sess_emb_hgnn, sess_emb_lgcn)
    neg1 = score(sess_emb_hgnn, row_column_shuffle(sess_emb_lgcn))
    one = torch.ones((neg1.shape[0])).cuda()

    con_loss = torch.sum(-torch.log(1e-7 + torch.sigmoid(pos)) - torch.log(1e-7 + (one - torch.sigmoid(neg1))))
    return con_loss
