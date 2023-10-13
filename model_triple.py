import numpy as np
import torch
from collections import defaultdict
import torch.nn.functional as F


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


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, category, category_emb, User_Item, Item_User, category_contain, args):
        super(SASRec, self).__init__()

        self.user_num = user_num + 1
        self.item_num = item_num + 1
        self.category = category
        self.category_num = 1534  # 704 707 518 1303 950
        self.max_category_len = 6
        self.dev = args.device
        self.User_Item = User_Item
        self.Item_User = Item_User
        self.category_contain = category_contain
        self.max_neighbor_len = 10

        self.item_emb = torch.nn.Embedding(self.item_num, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num, args.hidden_units, padding_idx=0)
        # self.category_emb = torch.nn.Embedding(self.category_num+1, args.hidden_units, padding_idx=0)
        self.category_emb = self.get_category_emb(category_emb, args)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.cross_attention_layernorms = torch.nn.ModuleList()
        self.cross_attention_layers = torch.nn.ModuleList()
        self.cross_forward_layernorms = torch.nn.ModuleList()
        self.cross_forward_layers = torch.nn.ModuleList()

        self.cross_attention_layernorms3 = torch.nn.ModuleList()
        self.cross_attention_layers3 = torch.nn.ModuleList()
        self.cross_forward_layernorms3 = torch.nn.ModuleList()
        self.cross_forward_layers3 = torch.nn.ModuleList()

        self.ca_attention_layernorms = torch.nn.ModuleList()
        self.ca_attention_layers = torch.nn.ModuleList()
        self.ca_forward_layernorms = torch.nn.ModuleList()
        self.ca_forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.last_cross_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.last_cross_layernorm3 = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.ca_last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.ic_last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.lc_last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.lc_last_layernorm2 = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.lc_last_layernorm3 = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        self.iui_last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.iui_attention_layernorms = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.cic_last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.cic_attention_layernorms = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.uiu_last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.uiu_attention_layernorms = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.uiciu_last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.uiciu_attention_layernorms = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.ciuic_last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.ciuic_attention_layernorms = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.ici_last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.ici_attention_layernorms = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units,
                                                         args.num_heads,
                                                         args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_cross_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.cross_attention_layernorms.append(new_cross_attn_layernorm)
            new_cross_attn_layer = torch.nn.MultiheadAttention(args.hidden_units,
                                                               args.num_heads,
                                                               args.dropout_rate)
            self.cross_attention_layers.append(new_cross_attn_layer)

            new_cross_attn_layernorm3 = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.cross_attention_layernorms3.append(new_cross_attn_layernorm3)
            new_cross_attn_layer3 = torch.nn.MultiheadAttention(args.hidden_units,
                                                                args.num_heads,
                                                                args.dropout_rate)
            self.cross_attention_layers3.append(new_cross_attn_layer3)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            new_cross_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.cross_forward_layernorms.append(new_cross_fwd_layernorm)
            new_cross_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.cross_forward_layers.append(new_cross_fwd_layer)

            new_cross_fwd_layernorm3 = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.cross_forward_layernorms3.append(new_cross_fwd_layernorm3)
            new_cross_fwd_layer3 = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.cross_forward_layers3.append(new_cross_fwd_layer3)

            ca_new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.ca_attention_layernorms.append(ca_new_attn_layernorm)

            ca_new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.ca_attention_layers.append(ca_new_attn_layer)

            ca_new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.ca_forward_layernorms.append(ca_new_fwd_layernorm)

            ca_new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.ca_forward_layers.append(ca_new_fwd_layer)

            self.ic_attention_layernorms = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

            self.ic_attention_layers = torch.nn.MultiheadAttention(args.hidden_units,
                                                                   args.num_heads,
                                                                   args.dropout_rate)

            self.ic_forward_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.ic_forward_layers = PointWiseFeedForward(args.hidden_units, args.dropout_rate)

        self.last_user_cross_layernorm = torch.nn.LayerNorm(4 * args.hidden_units, eps=1e-8)
        self.dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.gating = torch.nn.Linear(4 * args.hidden_units, args.hidden_units)

        self.last_user_cross_layernorm2 = torch.nn.LayerNorm(4 * args.hidden_units, eps=1e-8)
        self.dropout2 = torch.nn.Dropout(p=args.dropout_rate)
        # self.gating = torch.nn.Linear(args.hidden_units, args.hidden_units)
        self.gating2 = torch.nn.Linear(4 * args.hidden_units, args.hidden_units)

        self.last_user_cross_layernorm3 = torch.nn.LayerNorm(4 * args.hidden_units, eps=1e-8)
        self.dropout3 = torch.nn.Dropout(p=args.dropout_rate)
        self.gating3 = torch.nn.Linear(4 * args.hidden_units, args.hidden_units)

        self.last_user_cross_layernorm4 = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.dropout4 = torch.nn.Dropout(p=args.dropout_rate)
        self.gating4 = torch.nn.Linear(4 * args.hidden_units, args.hidden_units)

        self.last_user_cross_layernorm5 = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.dropout5 = torch.nn.Dropout(p=args.dropout_rate)
        self.gating5 = torch.nn.Linear(4 * args.hidden_units, args.hidden_units)

        self.last_user_cross_layernorm6 = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.dropout6 = torch.nn.Dropout(p=args.dropout_rate)
        self.gating6 = torch.nn.Linear(4 * args.hidden_units, args.hidden_units)

        self.UIU_adj = self.get_UIU_adj()
        self.UICIU_adj = self.get_UICIU_adj()

        self.CIC_adj = self.get_CIC_adj()
        self.CIUIC_adj = self.get_CIUIC_adj()

        self.IUI_adj = self.get_IUI_adj()
        self.ICI_adj = self.get_ICI_adj()

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
        for user in range(1, self.user_num):
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
        for user in range(1, self.user_num):
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
        for c in range(1, self.category_num):
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
        for c in range(1, self.category_num):
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
        for item in range(1, self.item_num):
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
        Adj = {}
        k = self.max_neighbor_len

        for item in range(1, self.item_num):
            item_count = {}
            item_categories = self.category[item]
            if item_categories[0] == 0:
                Adj[item] = [0] * k
                continue
            for category in item_categories:
                neighbor_items = self.category_contain[category]
                for neighbor_item in neighbor_items:
                    if neighbor_item in item_count:
                        item_count[neighbor_item] += 1
                    else:
                        item_count[neighbor_item] = 1

            sorted_items = sorted(item_count.items(), key=lambda x: x[1], reverse=True)
            Adj[item] = [sorted_items[i][0] for i in range(min(k, len(sorted_items)))]
        return Adj

    def log2feats(self, user_ids, log_seqs, log_seqs2, log_seqs3, mask2, mask3):
        seqsi1 = self.get_item_metapath_embedding_IUI(log_seqs)
        seqsi2 = self.get_item_metapath_embedding_ICI(log_seqs)

        seqs = seqsi1 + seqsi2

        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                                      attn_mask=attention_mask)

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        con_loss1 = SSL(seqsi1, seqsi2)

        seqsi3 = self.get_item_metapath_embedding_IUI(log_seqs2)
        seqsi4 = self.get_item_metapath_embedding_ICI(log_seqs2)
        seqs2 = seqsi3 + seqsi4
        seqs2 *= self.item_emb.embedding_dim ** 0.5
        positions2 = np.tile(np.array(range(log_seqs2.shape[1])), [log_seqs2.shape[0], 1])
        seqs2 += self.pos_emb(torch.LongTensor(positions2).to(self.dev))
        seqs2 = self.emb_dropout(seqs2)

        timeline_mask2 = torch.BoolTensor(log_seqs2 == 0).to(self.dev)
        seqs2 *= ~timeline_mask2.unsqueeze(-1)
        tl2 = seqs2.shape[1]
        batch_size = seqs2.shape[0]

        attention_mask2 = np.ones((batch_size, tl2, tl2), dtype=bool)
        for b in range(batch_size):
            for i in range(tl2):
                attention_mask2[b][i][0:mask2[b][i]] = False
        attention_mask2[:, :, 0] = False

        seqs2_np = seqs2.cpu().detach().numpy()
        att_seq2_np = seqs2_np
        for b in range(batch_size):
            for i in range(tl2):
                att_seq2_np[b][i] = seqs2_np[b][mask2[b][i] - 1]

        attention_mask2 = torch.from_numpy(attention_mask2).to(self.dev)
        att_seq2 = torch.from_numpy(att_seq2_np).to(self.dev)

        for i in range(len(self.cross_attention_layers)):
            att_seq2 = torch.transpose(att_seq2, 0, 1)
            seqs2 = torch.transpose(seqs2, 0, 1)

            Q2 = self.cross_attention_layernorms[i](att_seq2)

            mha_outputs2, _ = self.cross_attention_layers[i](Q2, seqs2, seqs2,
                                                             attn_mask=attention_mask2)
            att_seq2 = Q2 + mha_outputs2

            seqs2 = torch.transpose(seqs2, 0, 1)
            att_seq2 = torch.transpose(att_seq2, 0, 1)

            att_seq2 = self.cross_forward_layernorms[i](att_seq2)
            att_seq2 = self.cross_forward_layers[i](att_seq2)
            att_seq2 *= ~timeline_mask.unsqueeze(-1)  # mask2

        seqsi5 = self.get_item_metapath_embedding_IUI(log_seqs3)
        seqsi6 = self.get_item_metapath_embedding_ICI(log_seqs3)
        seqs3 = seqsi5 + seqsi6
        seqs3 *= self.item_emb.embedding_dim ** 0.5
        positions3 = np.tile(np.array(range(log_seqs3.shape[1])), [log_seqs3.shape[0], 1])
        seqs3 += self.pos_emb(torch.LongTensor(positions3).to(self.dev))
        seqs3 = self.emb_dropout(seqs3)

        timeline_mask3 = torch.BoolTensor(log_seqs3 == 0).to(self.dev)
        seqs3 *= ~timeline_mask3.unsqueeze(-1)
        tl3 = seqs3.shape[1]

        attention_mask3 = np.ones((batch_size, tl3, tl3), dtype=bool)
        for b in range(batch_size):
            for i in range(tl3):
                attention_mask3[b][i][0:mask3[b][i]] = False
        attention_mask3[:, :, 0] = False

        seqs3_np = seqs3.cpu().detach().numpy()
        att_seq3_np = seqs3_np
        for b in range(batch_size):
            for i in range(tl3):
                att_seq3_np[b][i] = seqs3_np[b][mask3[b][i] - 1]

        attention_mask3 = torch.from_numpy(attention_mask3).to(self.dev)
        att_seq3 = torch.from_numpy(att_seq3_np).to(self.dev)

        for i in range(len(self.cross_attention_layers3)):
            att_seq3 = torch.transpose(att_seq3, 0, 1)
            seqs3 = torch.transpose(seqs3, 0, 1)

            Q3 = self.cross_attention_layernorms3[i](att_seq3)

            mha_outputs3, _ = self.cross_attention_layers3[i](Q3, seqs3, seqs3,
                                                              attn_mask=attention_mask3)
            att_seq3 = Q3 + mha_outputs3

            seqs3 = torch.transpose(seqs3, 0, 1)
            att_seq3 = torch.transpose(att_seq3, 0, 1)

            att_seq3 = self.cross_forward_layernorms3[i](att_seq3)
            att_seq3 = self.cross_forward_layers3[i](att_seq3)
            att_seq3 *= ~timeline_mask.unsqueeze(-1)  # mask3

        c1 = self.get_category_metapath_embedding_CIC()  # C,H
        c2 = self.get_category_metapath_embedding_CIUIC()
        con_loss2 = SSL2(c1, c2)

        self.category_emb.weight = torch.nn.Parameter(c1 + c2)

        category_features = self.get_item_category_att(log_seqs)
        att_category_features = self.category_attention(log_seqs, category_features)

        category_features2 = self.get_item_category_att(log_seqs2)
        att_category_features2 = self.category_attention(log_seqs2, category_features2)

        category_features3 = self.get_item_category_att(log_seqs3)
        att_category_features3 = self.category_attention(log_seqs3, category_features3)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C) # LN (batch, len, hidden_size)
        log_feats2 = self.last_cross_layernorm(att_seq2)
        log_feats3 = self.last_cross_layernorm3(att_seq3)

        u1 = self.get_user_metapath_embedding_UIU(user_ids)  # B,H
        u2 = self.get_user_metapath_embedding_UICIU(user_ids)
        con_loss3 = SSL2(u1, u2)
        u = u1 + u2

        u *= self.user_emb.embedding_dim ** 0.5
        u = self.emb_dropout(u)

        u = u.unsqueeze(1)
        u = u.expand_as(seqs)

        cross = self.last_user_cross_layernorm4(self.dropout4(
            self.gating4(torch.cat((log_feats2, log_feats3, att_category_features2, att_category_features3), dim=2))))

        user = torch.cat((log_feats, att_category_features, cross, u), dim=2)
        user = self.last_user_cross_layernorm(user)
        user = self.dropout(self.gating(user))

        user = self.lc_last_layernorm(user)

        return user, con_loss1, con_loss2, con_loss3

    def log2feats2(self, user_ids, log_seqs, log_seqs2, log_seqs3, mask1, mask3):
        seqsi1 = self.get_item_metapath_embedding_IUI(log_seqs)
        seqsi2 = self.get_item_metapath_embedding_ICI(log_seqs)
        seqs = seqsi1 + seqsi2
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        seqsi3 = self.get_item_metapath_embedding_IUI(log_seqs2)
        seqsi4 = self.get_item_metapath_embedding_ICI(log_seqs2)
        con_loss1 = SSL(seqsi3, seqsi4)
        seqs2 = seqsi3 + seqsi4
        seqs2 *= self.item_emb.embedding_dim ** 0.5
        positions2 = np.tile(np.array(range(log_seqs2.shape[1])), [log_seqs2.shape[0], 1])
        seqs2 += self.pos_emb(torch.LongTensor(positions2).to(self.dev))
        seqs2 = self.emb_dropout(seqs2)

        timeline_mask2 = torch.BoolTensor(log_seqs2 == 0).to(self.dev)
        seqs2 *= ~timeline_mask2.unsqueeze(-1)

        tl2 = seqs2.shape[1]
        attention_mask2 = ~torch.tril(torch.ones((tl2, tl2), dtype=torch.bool, device=self.dev))

        for i in range(len(self.cross_attention_layers)):
            seqs2 = torch.transpose(seqs2, 0, 1)
            Q2 = self.cross_attention_layernorms[i](seqs2)
            mha_outputs2, _ = self.cross_attention_layers[i](Q2, seqs2, seqs2,
                                                             attn_mask=attention_mask2)

            seqs2 = Q2 + mha_outputs2
            seqs2 = torch.transpose(seqs2, 0, 1)

            seqs2 = self.cross_forward_layernorms[i](seqs2)
            seqs2 = self.cross_forward_layers[i](seqs2)
            seqs2 *= ~timeline_mask2.unsqueeze(-1)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)
        tl = seqs.shape[1]
        batch_size = seqs.shape[0]

        attention_mask = np.ones((batch_size, tl, tl), dtype=bool)
        for b in range(batch_size):
            for i in range(tl):
                attention_mask[b][i][0:mask1[b][i]] = False
        attention_mask[:, :, 0] = False

        seqs_np = seqs.cpu().detach().numpy()
        att_seq_np = seqs_np
        for b in range(batch_size):
            for i in range(tl):
                att_seq_np[b][i] = seqs_np[b][mask1[b][i] - 1]

        attention_mask = torch.from_numpy(attention_mask).to(self.dev)
        att_seq = torch.from_numpy(att_seq_np).to(self.dev)

        for i in range(len(self.attention_layers)):
            att_seq = torch.transpose(att_seq, 0, 1)
            seqs = torch.transpose(seqs, 0, 1)

            Q = self.attention_layernorms[i](att_seq)

            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                                      attn_mask=attention_mask)
            att_seq = Q + mha_outputs

            seqs = torch.transpose(seqs, 0, 1)
            att_seq = torch.transpose(att_seq, 0, 1)

            att_seq = self.forward_layernorms[i](att_seq)
            att_seq = self.forward_layers[i](att_seq)
            att_seq *= ~timeline_mask2.unsqueeze(-1)

        seqsi5 = self.get_item_metapath_embedding_IUI(log_seqs3)
        seqsi6 = self.get_item_metapath_embedding_ICI(log_seqs3)
        seqs3 = seqsi5 + seqsi6
        seqs3 *= self.item_emb.embedding_dim ** 0.5
        positions3 = np.tile(np.array(range(log_seqs3.shape[1])), [log_seqs3.shape[0], 1])
        seqs3 += self.pos_emb(torch.LongTensor(positions3).to(self.dev))
        seqs3 = self.emb_dropout(seqs3)

        timeline_mask3 = torch.BoolTensor(log_seqs3 == 0).to(self.dev)
        seqs3 *= ~timeline_mask3.unsqueeze(-1)
        tl3 = seqs3.shape[1]

        attention_mask3 = np.ones((batch_size, tl3, tl3), dtype=bool)
        for b in range(batch_size):
            for i in range(tl3):
                attention_mask3[b][i][0:mask3[b][i]] = False
        attention_mask3[:, :, 0] = False

        seqs3_np = seqs3.cpu().detach().numpy()
        att_seq3_np = seqs3_np
        for b in range(batch_size):
            for i in range(tl3):
                att_seq3_np[b][i] = seqs3_np[b][mask3[b][i] - 1]

        attention_mask3 = torch.from_numpy(attention_mask3).to(self.dev)
        att_seq3 = torch.from_numpy(att_seq3_np).to(self.dev)

        for i in range(len(self.cross_attention_layers3)):
            att_seq3 = torch.transpose(att_seq3, 0, 1)
            seqs3 = torch.transpose(seqs3, 0, 1)

            Q3 = self.cross_attention_layernorms3[i](att_seq3)

            mha_outputs3, _ = self.cross_attention_layers3[i](Q3, seqs3, seqs3,
                                                              attn_mask=attention_mask3)
            att_seq3 = Q3 + mha_outputs3

            seqs3 = torch.transpose(seqs3, 0, 1)
            att_seq3 = torch.transpose(att_seq3, 0, 1)

            att_seq3 = self.cross_forward_layernorms3[i](att_seq3)
            att_seq3 = self.cross_forward_layers3[i](att_seq3)
            att_seq3 *= ~timeline_mask.unsqueeze(-1)

        c1 = self.get_category_metapath_embedding_CIC()
        c2 = self.get_category_metapath_embedding_CIUIC()
        con_loss2 = SSL2(c1, c2)

        self.category_emb.weight = torch.nn.Parameter(c1 + c2)

        category_features = self.get_item_category_att(log_seqs)
        att_category_features = self.category_attention(log_seqs, category_features)

        category_features2 = self.get_item_category_att(log_seqs2)
        att_category_features2 = self.category_attention(log_seqs2, category_features2)

        category_features3 = self.get_item_category_att(log_seqs3)
        att_category_features3 = self.category_attention(log_seqs3, category_features3)

        log_feats = self.last_layernorm(att_seq)
        log_feats2 = self.last_cross_layernorm(seqs2)
        log_feats3 = self.last_cross_layernorm3(att_seq3)

        u1 = self.get_user_metapath_embedding_UIU(user_ids)
        u2 = self.get_user_metapath_embedding_UICIU(user_ids)
        con_loss3 = SSL2(u1, u2)
        u = u1 + u2

        u *= self.user_emb.embedding_dim ** 0.5
        u = self.emb_dropout(u)

        u = u.unsqueeze(1)
        u = u.expand_as(seqs)

        cross = self.last_user_cross_layernorm5(self.dropout5(
            self.gating5(torch.cat((log_feats, log_feats3, att_category_features, att_category_features3), dim=2))))

        user = torch.cat((log_feats2, att_category_features2, cross, u), dim=2)
        user = self.last_user_cross_layernorm2(user)
        user = self.dropout2(self.gating2(user))

        user = self.lc_last_layernorm2(user)

        return user, con_loss1, con_loss2, con_loss3

    def log2feats3(self, user_ids, log_seqs, log_seqs2, log_seqs3, mask1, mask2):
        seqsi1 = self.get_item_metapath_embedding_IUI(log_seqs)
        seqsi2 = self.get_item_metapath_embedding_ICI(log_seqs)
        seqs = seqsi1 + seqsi2
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        seqsi5 = self.get_item_metapath_embedding_IUI(log_seqs3)
        seqsi6 = self.get_item_metapath_embedding_ICI(log_seqs3)
        seqs3 = seqsi5 + seqsi6
        positions3 = np.tile(np.array(range(log_seqs3.shape[1])), [log_seqs3.shape[0], 1])
        seqs3 += self.pos_emb(torch.LongTensor(positions3).to(self.dev))
        seqs3 = self.emb_dropout(seqs3)

        con_loss1 = SSL(seqsi5, seqsi6)

        timeline_mask3 = torch.BoolTensor(log_seqs3 == 0).to(self.dev)
        seqs3 *= ~timeline_mask3.unsqueeze(-1)

        tl3 = seqs3.shape[1]
        attention_mask3 = ~torch.tril(torch.ones((tl3, tl3), dtype=torch.bool, device=self.dev))

        for i in range(len(self.cross_attention_layers3)):
            seqs3 = torch.transpose(seqs3, 0, 1)
            Q3 = self.cross_attention_layernorms3[i](seqs3)
            mha_outputs3, _ = self.cross_attention_layers3[i](Q3, seqs3, seqs3,
                                                              attn_mask=attention_mask3)

            seqs3 = Q3 + mha_outputs3
            seqs3 = torch.transpose(seqs3, 0, 1)

            seqs3 = self.cross_forward_layernorms3[i](seqs3)
            seqs3 = self.cross_forward_layers3[i](seqs3)
            seqs3 *= ~timeline_mask3.unsqueeze(-1)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)
        tl = seqs.shape[1]
        batch_size = seqs.shape[0]

        attention_mask = np.ones((batch_size, tl, tl), dtype=bool)
        for b in range(batch_size):
            for i in range(tl):
                attention_mask[b][i][0:mask1[b][i]] = False
        attention_mask[:, :, 0] = False

        seqs_np = seqs.cpu().detach().numpy()
        att_seq_np = seqs_np
        for b in range(batch_size):
            for i in range(tl):
                att_seq_np[b][i] = seqs_np[b][mask1[b][i] - 1]

        attention_mask = torch.from_numpy(attention_mask).to(self.dev)
        att_seq = torch.from_numpy(att_seq_np).to(self.dev)

        for i in range(len(self.attention_layers)):
            att_seq = torch.transpose(att_seq, 0, 1)
            seqs = torch.transpose(seqs, 0, 1)

            Q = self.attention_layernorms[i](att_seq)

            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                                      attn_mask=attention_mask)
            att_seq = Q + mha_outputs

            seqs = torch.transpose(seqs, 0, 1)
            att_seq = torch.transpose(att_seq, 0, 1)

            att_seq = self.forward_layernorms[i](att_seq)
            att_seq = self.forward_layers[i](att_seq)
            att_seq *= ~timeline_mask3.unsqueeze(-1)

        seqsi3 = self.get_item_metapath_embedding_IUI(log_seqs2)
        seqsi4 = self.get_item_metapath_embedding_ICI(log_seqs2)
        seqs2 = seqsi3 + seqsi4
        seqs2 *= self.item_emb.embedding_dim ** 0.5
        positions2 = np.tile(np.array(range(log_seqs2.shape[1])), [log_seqs2.shape[0], 1])
        seqs2 += self.pos_emb(torch.LongTensor(positions2).to(self.dev))
        seqs2 = self.emb_dropout(seqs2)

        timeline_mask2 = torch.BoolTensor(log_seqs2 == 0).to(self.dev)
        seqs2 *= ~timeline_mask2.unsqueeze(-1)
        tl2 = seqs2.shape[1]

        attention_mask2 = np.ones((batch_size, tl2, tl2), dtype=bool)
        for b in range(batch_size):
            for i in range(tl2):
                attention_mask2[b][i][0:mask2[b][i]] = False
        attention_mask2[:, :, 0] = False

        seqs2_np = seqs2.cpu().detach().numpy()
        att_seq2_np = seqs2_np
        for b in range(batch_size):
            for i in range(tl2):
                att_seq2_np[b][i] = seqs2_np[b][mask2[b][i] - 1]

        attention_mask2 = torch.from_numpy(attention_mask2).to(self.dev)
        att_seq2 = torch.from_numpy(att_seq2_np).to(self.dev)

        for i in range(len(self.cross_attention_layers)):
            att_seq2 = torch.transpose(att_seq2, 0, 1)
            seqs2 = torch.transpose(seqs2, 0, 1)

            Q2 = self.cross_attention_layernorms[i](att_seq2)

            mha_outputs2, _ = self.cross_attention_layers[i](Q2, seqs2, seqs2,
                                                             attn_mask=attention_mask2)
            att_seq2 = Q2 + mha_outputs2

            seqs2 = torch.transpose(seqs2, 0, 1)
            att_seq2 = torch.transpose(att_seq2, 0, 1)

            att_seq2 = self.cross_forward_layernorms[i](att_seq2)
            att_seq2 = self.cross_forward_layers[i](att_seq2)
            att_seq2 *= ~timeline_mask.unsqueeze(-1)  # mask3

        c1 = self.get_category_metapath_embedding_CIC()
        c2 = self.get_category_metapath_embedding_CIUIC()
        con_loss2 = SSL2(c1, c2)

        self.category_emb.weight = torch.nn.Parameter(c1 + c2)

        category_features = self.get_item_category_att(log_seqs)
        att_category_features = self.category_attention(log_seqs, category_features)

        category_features2 = self.get_item_category_att(log_seqs2)
        att_category_features2 = self.category_attention(log_seqs2, category_features2)

        category_features3 = self.get_item_category_att(log_seqs3)
        att_category_features3 = self.category_attention(log_seqs3, category_features3)

        log_feats = self.last_layernorm(att_seq)
        log_feats2 = self.last_cross_layernorm(att_seq2)
        log_feats3 = self.last_cross_layernorm3(seqs3)

        u1 = self.get_user_metapath_embedding_UIU(user_ids)
        u2 = self.get_user_metapath_embedding_UICIU(user_ids)
        con_loss3 = SSL2(u1, u2)
        u = u1 + u2

        u *= self.user_emb.embedding_dim ** 0.5
        u = self.emb_dropout(u)

        u = u.unsqueeze(1)
        u = u.expand_as(seqs)

        cross = self.last_user_cross_layernorm6(self.dropout6(
            self.gating6(torch.cat((log_feats, log_feats2, att_category_features, att_category_features2), dim=2))))

        user = torch.cat((log_feats3, att_category_features3, cross, u), dim=2)
        user = self.last_user_cross_layernorm3(user)
        user = self.dropout3(self.gating3(user))  # 拼接

        user = self.lc_last_layernorm3(user)

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
            log_feats, con_loss1, con_loss2, con_loss3 = self.log2feats(user_ids, log_seqs, log_seqs2, log_seqs3, mask1,
                                                                        mask2)
        elif target == "B":
            log_feats, con_loss1, con_loss2, con_loss3 = self.log2feats2(user_ids, log_seqs, log_seqs2, log_seqs3,
                                                                         mask1, mask2)
        else:
            log_feats, con_loss1, con_loss2, con_loss3 = self.log2feats3(user_ids, log_seqs, log_seqs2, log_seqs3,
                                                                         mask1, mask2)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)


        return pos_logits, neg_logits, con_loss1 / (
                    log_feats.shape[0] * log_feats.shape[1]), con_loss2 / self.category_num, con_loss3 / \
               log_feats.shape[0]

    def predict(self, user_ids, log_seqs, log_seqs2, log_seqs3, item_indices, mask1, mask2, target):
        if target == "A":
            log_feats, con_loss1, con_loss2, con_loss3 = self.log2feats(user_ids, log_seqs, log_seqs2, log_seqs3, mask1,
                                                                        mask2)
        elif target == "B":
            log_feats, con_loss1, con_loss2, con_loss3 = self.log2feats2(user_ids, log_seqs, log_seqs2, log_seqs3,
                                                                         mask1,
                                                                         mask2)
        else:
            log_feats, con_loss1, con_loss2, con_loss3 = self.log2feats3(user_ids, log_seqs, log_seqs2, log_seqs3,
                                                                         mask1,
                                                                         mask2)

        final_feat = log_feats[:, -1, :]

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits


def norm(adj):
    adj += np.eye(adj.shape[0])
    degree = np.array(adj.sum(1))
    degree = np.diag(np.power(degree, -0.5))
    return degree.dot(adj).dot(degree)


class GraphConvolution(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(GraphConvolution, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, adj, features):
        out = torch.mm(adj, features)
        out = self.linear(out)
        return out


def SSL(sess_emb_hgnn, sess_emb_lgcn):
    def row_shuffle(embedding):
        corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
        return corrupted_embedding

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


def SSL2(sess_emb_hgnn, sess_emb_lgcn):
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


class GCN(torch.nn.Module):
    def __init__(self, input_size=50, hidden_size=50):
        super(GCN, self).__init__()
        self.relu = torch.nn.ReLU()
        self.gcn1 = GraphConvolution(input_size, hidden_size)
        self.gcn2 = GraphConvolution(hidden_size, 50)

    def forward(self, adj, features):
        out = self.relu(self.gcn1(adj, features))
        out = self.gcn2(adj, out)
        return out

    def get_embedding(self, log_seqs):
        return self.item_all_embedding[log_seqs]
