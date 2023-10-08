import numpy as np
import torch
from collections import defaultdict
import torch.nn.functional as F


class PointWiseFeedForward(torch.nn.Module):  # 前馈层（使用两层CNN）
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs  # 残差
        return outputs


# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, category, category_emb, User_Item, Item_User, category_contain, args):
        super(SASRec, self).__init__()

        self.user_num = user_num + 1
        self.item_num = item_num + 1
        self.category = category
        self.category_num = 1534 # 704 707 518 1303 950
        self.max_category_len = 6
        self.dev = args.device
        self.User_Item = User_Item
        self.Item_User = Item_User
        self.category_contain = category_contain
        self.max_neighbor_len = 10

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num, args.hidden_units, padding_idx=0)
        # self.category_emb = torch.nn.Embedding(self.category_num+1, args.hidden_units, padding_idx=0)
        self.category_emb = self.get_category_emb(category_emb, args)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)  # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
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

        self.ca_attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
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


        for _ in range(args.num_blocks):  # 每个块
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units,
                                                         args.num_heads,
                                                         args.dropout_rate)  # attention
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
            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)  # 前馈网络
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
                                                            args.dropout_rate)  # attention
            self.ca_attention_layers.append(ca_new_attn_layer)

            ca_new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.ca_forward_layernorms.append(ca_new_fwd_layernorm)

            ca_new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)  # 前馈网络
            self.ca_forward_layers.append(ca_new_fwd_layer)

            self.ic_attention_layernorms = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

            self.ic_attention_layers = torch.nn.MultiheadAttention(args.hidden_units,
                                                                   args.num_heads,
                                                                   args.dropout_rate)  # attention

            self.ic_forward_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.ic_forward_layers = PointWiseFeedForward(args.hidden_units, args.dropout_rate)  # 前馈网络

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

        self.last_user_cross_layernorm5 = torch.nn.LayerNorm( args.hidden_units, eps=1e-8)
        self.dropout5 = torch.nn.Dropout(p=args.dropout_rate)
        self.gating5 = torch.nn.Linear(4 * args.hidden_units, args.hidden_units)

        self.last_user_cross_layernorm6 = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.dropout6 = torch.nn.Dropout(p=args.dropout_rate)
        self.gating6 = torch.nn.Linear(4 * args.hidden_units, args.hidden_units)



        # self.item_agg_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        # self.item_agg_gating = torch.nn.Linear(2 * args.hidden_units, args.hidden_units)
        # self.item2_agg_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        # self.item2_agg_gating = torch.nn.Linear(2 * args.hidden_units, args.hidden_units)
        # self.user_agg_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        # self.user_agg_gating = torch.nn.Linear(2 * args.hidden_units, args.hidden_units)
        # self.category_agg_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        # self.category_agg_gating = torch.nn.Linear(2 * args.hidden_units, args.hidden_units)

        # self.GCN = GCN(args.hidden_units, args.hidden_units).to(self.dev)
        # self.category_adj = self.get_category_adj().to(self.dev)

        # self.UIU_adj = self.get_UIU_adj(User_Item, Item_User).to(self.dev)
        # self.UICIU_adj = self.get_UICIU_adj(User_Item, Item_User, category_contain).to(self.dev)
        # self.UIU_GCN = GCN(args.hidden_units, args.hidden_units).to(self.dev)
        # self.UICIU_GCN = GCN(args.hidden_units, args.hidden_units).to(self.dev)
        #
        # self.CIC_adj = self.get_CIC_adj(category_contain).to(self.dev)
        # self.CIUIC_adj = self.get_CIUIC_adj(User_Item, Item_User, category_contain).to(self.dev)
        # self.CIC_GCN = GCN(args.hidden_units, args.hidden_units).to(self.dev)
        # self.CIUIC_GCN = GCN(args.hidden_units, args.hidden_units).to(self.dev)

        self.UIU_adj = self.get_UIU_adj()
        self.UICIU_adj = self.get_UICIU_adj()

        self.CIC_adj = self.get_CIC_adj()
        self.CIUIC_adj = self.get_CIUIC_adj()

        self.IUI_adj = self.get_IUI_adj()
        self.ICI_adj = self.get_ICI_adj()

        # self.ICI_adj = self.get_ICI_adj(User_Item, Item_User, category_contain).to(self.dev)
        # self.IUI_GCN = GCN(args.hidden_units, args.hidden_units).to(self.dev)

    #         self.ICI_GCN = GCN(args.hidden_units, args.hidden_units).to(self.dev)

    def get_category_emb(self, category_emb, args):
        embedding_layer = torch.nn.Embedding(self.category_num, args.hidden_units, padding_idx=0)
        linear_layer = torch.nn.Linear(768, args.hidden_units)
        category_emb = linear_layer(category_emb)
        embedding_layer.weight = torch.nn.Parameter(category_emb)
        return embedding_layer

    def get_category_adj(self):
        A = np.zeros((self.category_num, self.category_num))  # 邻接矩阵
        N = np.zeros(self.category_num)  # ci出现次数
        M = np.zeros((self.category_num, self.category_num))  # ci和cj同时出现的次数

        for values in self.category.values():
            for idx, v in enumerate(values):  # 遍历list
                N[v] += 1
                for j in range(idx + 1, len(values)):  # 比较i和list中i后面的每个元素
                    if values[j] != 0:
                        M[v][values[j]] += 1
        # print(np.sort(N)[-10:])
        # print(np.sort(M.flatten())[-10:])
        # for i, val in enumerate(np.sort(M.flatten())[-10:]):
        #     max_indices = np.where(M == val)
        #     row, col = max_indices[0][0], max_indices[1][0]
        #     print(f"Element {i + 1}: {val} at position ({row}, {col})")

        for i in range(self.category_num):
            for j in range(self.category_num):
                A[i][j] = M[i][j] / N[i]

        indices = []  # 存储每行中大于0.5的元素的下标以及它们所在的行的下标
        for i, row in enumerate(A):
            row_indices = []  # 存储该行中大于0.5的元素的下标
            for j, val in enumerate(row):
                if val > 0.5:
                    row_indices.append(j)
                    indices.append([i, j])  # 将该元素的下标以及该行的下标存储到`indices`列表中
            if len(row_indices) > 0:  # 如果该行中有大于0.5的元素，打印该行的下标以及这些元素的下标
                print(f"Row {i}: {row_indices}")

        # print(indices)  # 输出所有大于0.5的元素的下标以及它们所在的行的下标

        # top05_means = []  # 存储每行大于0.5的元素的平均值
        # for row in A:
        #     filtered_row = list(filter(lambda x: x > 0.5, row))  # 筛选出大于0.5的元素
        #     if len(filtered_row) == 0:  # 如果该行中没有大于0.5的元素，跳过该行
        #         continue
        #     print(filtered_row)
        #     row_mean = sum(filtered_row) / len(filtered_row)  # 计算该行中大于0.5的元素的平均值
        #     top05_means.append(row_mean)
        #
        # total_mean = sum(top05_means) / len(top05_means)
        # print(total_mean, len(top05_means))
        #
        # total_mean = sum(top10_means) / len(top10_means)  # 所有值的平均值
        # print(total_mean) # 0.23

        # print(np.sort(A.flatten())[-300:])

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

        # A = np.zeros((self.user_num, self.user_num))  # 邻接矩阵
        # # 遍历所有用户
        # for i, user in enumerate(User_Item):
        #     # 遍历用户u交互过的物品
        #     for item in User_Item[user]:
        #         # 找到交互过该物品的所有用户
        #         # if item in Item_User:
        #         neighbors = Item_User[item]
        #         for neighbor in neighbors:
        #             A[user, neighbor] += 1  # 邻接矩阵上的值+1
        #             # if neighbor != user:
        #             #     A[i, neighbor - 1] += 1  # 邻接矩阵上的值+1
        #
        # # 进行加权处理
        # A[A == 0] = 1e-10
        # weighted_adj_matrix = A / np.sum(A, axis=1, keepdims=True)

        # return torch.Tensor(weighted_adj_matrix)

    # def get_UICIU_adj(self):
    #     Adj = {}
    #     k = 10
    #     for user in range(1, self.user_num):
    #         user_count = {}
    #         items = self.User_Item[user]
    #         for item in items:
    #             # neighbor_users = self.Item_User[item]
    #             categories = self.category[item]
    #             if categories[0] == 0:
    #                 continue
    #             for c in categories:
    #                 categories_items = self.category_contain[c]
    #                 # if len(categories_items) > 10:
    #                 #     categories_items = list(categories_items)[:10]
    #                 for categories_item in categories_items:
    #                     neighbor_users = self.Item_User[categories_item]
    #                     for neighbor_user in neighbor_users:
    #                         if neighbor_user in user_count:
    #                             user_count[neighbor_user] += 1
    #                         else:
    #                             user_count[neighbor_user] = 1
    #         sorted_users = sorted(user_count.items(), key=lambda x:x[1], reverse=True)
    #         Adj[user] = [sorted_users[i][0] for i in range(min(k, len(sorted_users)))]
    #     return Adj

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
                # if len(categories_items) > 100:
                #     categories_items = list(categories_items)[:100]
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

        # A = np.zeros((self.user_num, self.user_num))  # 邻接矩阵
        # # 遍历所有用户
        # for i, user in enumerate(User_Item):
        #     # 遍历用户u交互过的物品
        #     for item in User_Item[user]:
        #         # 找到物品所属的类别
        #         # if item in self.category:
        #         categories = self.category[item]
        #         if categories[0] == 0:  # 去除"NULL"
        #             continue
        #         for cat in categories:
        #             # 找到类别包含的物品
        #             # if cat in category_contain:
        #             items = category_contain[cat]
        #             if len(items) > 5:
        #                 items = list(items)[:5]
        #             for neighbor_item in items:
        #                 # 找到交互过该物品的所有用户
        #                 # if neighbor_item in Item_User:
        #                 neighbors = Item_User[neighbor_item]
        #                 for neighbor in neighbors:
        #                     A[user, neighbor] += 1  # 邻接矩阵上的值+1
        #                     # if neighbor != user:
        #                     #     A[i, neighbor - 1] += 1  # 邻接矩阵上的值+1
        #
        # # 进行加权处理
        # A[A == 0] = 1e-10
        # weighted_adj_matrix = A / np.sum(A, axis=1, keepdims=True)
        #
        # return torch.Tensor(weighted_adj_matrix)

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
                # neighbor_users = self.Item_User[item]
                users = self.Item_User[item]
                # if categories[0] == 0:
                #     continue
                for u in users:
                    if u in user_count:
                        user_count[u] += 1
                    else:
                        user_count[u] = 1
            sorted_users = sorted(user_count.items(), key=lambda x: x[1], reverse=True)
            top_users = [sorted_users[i][0] for i in range(min(k, len(sorted_users)))]
            for u in top_users:
                users_items = self.User_Item[u]
                # if len(categories_items) > 10:
                #     categories_items = list(categories_items)[:10]
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

    # def get_CIC_adj(self):
    #     A = np.zeros((self.category_num, self.category_num))  # 邻接矩阵
    #     # 遍历所有类别
    #     for c, items in category_contain.items():
    #         # 遍历类别c包含的物品
    #         if len(items) > 5:
    #             items = list(items)[:5]
    #         if c == 0:
    #             continue
    #         for item in items:
    #             # 找到物品所属的类别
    #             # if item in category:
    #             item_categories = self.category[item]
    #             for neighbor_category in item_categories:
    #                 A[c, neighbor_category] += 1  # 邻接矩阵上的值+1
    #                 # if neighbor != user:
    #                 #     A[i, neighbor - 1] += 1  # 邻接矩阵上的值+1
    #
    #     # 进行加权处理
    #     A[A == 0] = 1e-10
    #     weighted_adj_matrix = A / np.sum(A, axis=1, keepdims=True)
    #
    #     return torch.Tensor(weighted_adj_matrix)
    #
    # def get_CIUIC_adj(self):
    #     A = np.zeros((self.category_num, self.category_num))  # 邻接矩阵
    #     # 遍历所有类别
    #     for c, items in category_contain.items():
    #         if len(items) > 5:
    #             items = list(items)[:5]
    #         if c == 0:
    #             continue
    #         # 遍历类别c包含的物品
    #         for item in items:
    #             # 找到包含物品的用户
    #             # if item in Item_User:
    #             users = Item_User[item]
    #             for user in users:
    #                 # 查找用户交互过的物品
    #                 # if user in User_Item:
    #                 user_items = User_Item[user]
    #                 for neighbor_item in user_items:
    #                     # 查找物品对应的类别
    #                     # if neighbor_item in self.category:
    #                     neighbor_categories = self.category[neighbor_item]
    #                     if neighbor_categories[0] == 0:
    #                         continue
    #                     # 将这些类别视为c的邻居
    #                     for neighbor_category in neighbor_categories:
    #                         # if neighbor_category != c:
    #                         A[c, neighbor_category] += 1  # 邻接矩阵上的值+1
    #
    #     # 进行加权处理
    #     A[A == 0] = 1e-10
    #     weighted_adj_matrix = A / np.sum(A, axis=1, keepdims=True)
    #
    #     return torch.Tensor(weighted_adj_matrix)

    def get_IUI_adj(self):
        Adj = {}
        k = self.max_neighbor_len
        # Adj[0] = [0] * 10
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
        # Adj[0] = [0] * 10

        for item in range(1, self.item_num):
            item_count = {}
            item_categories = self.category[item]
            if item_categories[0] == 0:
                Adj[item] = [0] * k
                continue
            for category in item_categories:
                neighbor_items = self.category_contain[category]
                # if len(neighbor_items) > 100:
                #     neighbor_items = list(neighbor_items)[:100]
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
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))  # 加上位置向量
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):  # 子模块的每一个网络 ?为什么不直接用
            seqs = torch.transpose(seqs, 0, 1)  # 转置，为了符合attention输入
            Q = self.attention_layernorms[i](seqs)  # LN后的作为Q？
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,  # attention Q,K,V   # 测试发现全设置为Q影响也不大
                                                      attn_mask=attention_mask)

            seqs = Q + mha_outputs  # 残差
            seqs = torch.transpose(seqs, 0, 1)  # 转回来

            seqs = self.forward_layernorms[i](seqs)  # LN
            seqs = self.forward_layers[i](seqs)  # 前馈层
            seqs *= ~timeline_mask.unsqueeze(-1)

        con_loss1 = SSL(seqsi1, seqsi2)  # TODO: conclude log_seqs2

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
            att_seq2 *= ~timeline_mask.unsqueeze(-1) # mask2

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

        c1 = self.get_category_metapath_embedding_CIC() # C,H
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

        u1 = self.get_user_metapath_embedding_UIU(user_ids) # B,H
        u2 = self.get_user_metapath_embedding_UICIU(user_ids)
        con_loss3 = SSL2(u1, u2)
        u = u1 + u2

        u *= self.user_emb.embedding_dim ** 0.5
        u = self.emb_dropout(u)

        u = u.unsqueeze(1)
        u = u.expand_as(seqs)

        cross = self.last_user_cross_layernorm4(self.dropout4(self.gating4(torch.cat((log_feats2, log_feats3, att_category_features2, att_category_features3),dim=2))))
        # cross = self.last_user_cross_layernorm4(self.dropout4(self.gating4(log_feats2)))

        user = torch.cat((log_feats, att_category_features, cross, u), dim=2)
        user = self.last_user_cross_layernorm(user)
        user = self.dropout(self.gating(user))  # 拼接

        user = self.lc_last_layernorm(user)

        return user, con_loss1, con_loss2, con_loss3

    def log2feats2(self, user_ids, log_seqs, log_seqs2, log_seqs3, mask1, mask3):
        seqsi1 = self.get_item_metapath_embedding_IUI(log_seqs)
        seqsi2 = self.get_item_metapath_embedding_ICI(log_seqs)
        seqs = seqsi1 + seqsi2
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))  # 加上位置向量
        seqs = self.emb_dropout(seqs)

        # seqsi5 = self.get_item_metapath_embedding_IUI(log_seqs3)
        # seqsi6 = self.get_item_metapath_embedding_ICI(log_seqs3)
        # seqs3 = seqsi5 + seqsi6
        # positions3 = np.tile(np.array(range(log_seqs3.shape[1])), [log_seqs3.shape[0], 1])
        # seqs3 += self.pos_emb(torch.LongTensor(positions3).to(self.dev))  # 加上位置向量
        # seqs3 = self.emb_dropout(seqs3)

        seqsi3 = self.get_item_metapath_embedding_IUI(log_seqs2)
        seqsi4 = self.get_item_metapath_embedding_ICI(log_seqs2)
        con_loss1 = SSL(seqsi3, seqsi4)  # TODO: conclude log_seqs2
        seqs2 = seqsi3 + seqsi4
        seqs2 *= self.item_emb.embedding_dim ** 0.5
        positions2 = np.tile(np.array(range(log_seqs2.shape[1])), [log_seqs2.shape[0], 1])
        seqs2 += self.pos_emb(torch.LongTensor(positions2).to(self.dev))
        seqs2 = self.emb_dropout(seqs2)

        timeline_mask2 = torch.BoolTensor(log_seqs2 == 0).to(self.dev)
        seqs2 *= ~timeline_mask2.unsqueeze(-1)  # broadcast in last dim

        tl2 = seqs2.shape[1]  # time dim len for enforce causality
        attention_mask2 = ~torch.tril(torch.ones((tl2, tl2), dtype=torch.bool, device=self.dev))

        for i in range(len(self.cross_attention_layers)):  # 子模块的每一个网络 ?为什么不直接用
            seqs2 = torch.transpose(seqs2, 0, 1)  # 转置，为了符合attention输入
            Q2 = self.cross_attention_layernorms[i](seqs2)  # LN后的作为Q？
            mha_outputs2, _ = self.cross_attention_layers[i](Q2, seqs2, seqs2,  # attention Q,K,V   # 测试发现全设置为Q影响也不大
                                                      attn_mask=attention_mask2)

            seqs2 = Q2 + mha_outputs2  # 残差
            seqs2 = torch.transpose(seqs2, 0, 1)  # 转回来

            seqs2 = self.cross_forward_layernorms[i](seqs2)  # LN
            seqs2 = self.cross_forward_layers[i](seqs2)  # 前馈层
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
            att_seq *= ~timeline_mask2.unsqueeze(-1) # mask2

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


        c1 = self.get_category_metapath_embedding_CIC() # C,H
        c2 = self.get_category_metapath_embedding_CIUIC()
        con_loss2 = SSL2(c1, c2)

        self.category_emb.weight = torch.nn.Parameter(c1 + c2)

        category_features = self.get_item_category_att(log_seqs)
        att_category_features = self.category_attention(log_seqs, category_features)

        category_features2 = self.get_item_category_att(log_seqs2)
        att_category_features2 = self.category_attention(log_seqs2, category_features2)

        category_features3 = self.get_item_category_att(log_seqs3)
        att_category_features3 = self.category_attention(log_seqs3, category_features3)

        log_feats = self.last_layernorm(att_seq)  # (U, T, C) -> (U, -1, C) # LN (batch, len, hidden_size)
        log_feats2 = self.last_cross_layernorm(seqs2)
        log_feats3 = self.last_cross_layernorm3(att_seq3)

        u1 = self.get_user_metapath_embedding_UIU(user_ids) # B,H
        u2 = self.get_user_metapath_embedding_UICIU(user_ids)
        con_loss3 = SSL2(u1, u2)
        u = u1 + u2

        u *= self.user_emb.embedding_dim ** 0.5
        u = self.emb_dropout(u)

        u = u.unsqueeze(1)
        u = u.expand_as(seqs)

        cross = self.last_user_cross_layernorm5(self.dropout5(self.gating5(torch.cat((log_feats, log_feats3, att_category_features, att_category_features3),dim=2))))
        # cross = self.last_user_cross_layernorm3(self.dropout3(self.gating3(log_feats)))

        user = torch.cat((log_feats2, att_category_features2, cross, u), dim=2)
        user = self.last_user_cross_layernorm2(user)
        user = self.dropout2(self.gating2(user))  # 拼接

        user = self.lc_last_layernorm2(user)

        return user, con_loss1, con_loss2, con_loss3

    def log2feats3(self, user_ids, log_seqs, log_seqs2, log_seqs3, mask1, mask2):
        seqsi1 = self.get_item_metapath_embedding_IUI(log_seqs)
        seqsi2 = self.get_item_metapath_embedding_ICI(log_seqs)
        seqs = seqsi1 + seqsi2
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))  # 加上位置向量
        seqs = self.emb_dropout(seqs)

        seqsi5 = self.get_item_metapath_embedding_IUI(log_seqs3)
        seqsi6 = self.get_item_metapath_embedding_ICI(log_seqs3)
        seqs3 = seqsi5 + seqsi6
        positions3 = np.tile(np.array(range(log_seqs3.shape[1])), [log_seqs3.shape[0], 1])
        seqs3 += self.pos_emb(torch.LongTensor(positions3).to(self.dev))  # 加上位置向量
        seqs3 = self.emb_dropout(seqs3)

        con_loss1 = SSL(seqsi5, seqsi6)  # TODO: conclude log_seqs2


        timeline_mask3 = torch.BoolTensor(log_seqs3 == 0).to(self.dev)
        seqs3 *= ~timeline_mask3.unsqueeze(-1)  # broadcast in last dim

        tl3 = seqs3.shape[1]  # time dim len for enforce causality
        attention_mask3 = ~torch.tril(torch.ones((tl3, tl3), dtype=torch.bool, device=self.dev))

        for i in range(len(self.cross_attention_layers3)):  # 子模块的每一个网络 ?为什么不直接用
            seqs3 = torch.transpose(seqs3, 0, 1)  # 转置，为了符合attention输入
            Q3 = self.cross_attention_layernorms3[i](seqs3)  # LN后的作为Q？
            mha_outputs3, _ = self.cross_attention_layers3[i](Q3, seqs3, seqs3,  # attention Q,K,V   # 测试发现全设置为Q影响也不大
                                                      attn_mask=attention_mask3)

            seqs3 = Q3 + mha_outputs3  # 残差
            seqs3 = torch.transpose(seqs3, 0, 1)  # 转回来

            seqs3 = self.cross_forward_layernorms3[i](seqs3)  # LN
            seqs3 = self.cross_forward_layers3[i](seqs3)  # 前馈层
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
            att_seq *= ~timeline_mask3.unsqueeze(-1) # mask2

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


        c1 = self.get_category_metapath_embedding_CIC() # C,H
        c2 = self.get_category_metapath_embedding_CIUIC()
        con_loss2 = SSL2(c1, c2)

        self.category_emb.weight = torch.nn.Parameter(c1 + c2)

        category_features = self.get_item_category_att(log_seqs)
        att_category_features = self.category_attention(log_seqs, category_features)

        category_features2 = self.get_item_category_att(log_seqs2)
        att_category_features2 = self.category_attention(log_seqs2, category_features2)

        category_features3 = self.get_item_category_att(log_seqs3)
        att_category_features3 = self.category_attention(log_seqs3, category_features3)

        log_feats = self.last_layernorm(att_seq)  # (U, T, C) -> (U, -1, C) # LN (batch, len, hidden_size)
        log_feats2 = self.last_cross_layernorm(att_seq2)
        log_feats3 = self.last_cross_layernorm3(seqs3)

        u1 = self.get_user_metapath_embedding_UIU(user_ids) # B,H
        u2 = self.get_user_metapath_embedding_UICIU(user_ids)
        con_loss3 = SSL2(u1, u2)
        u = u1 + u2

        u *= self.user_emb.embedding_dim ** 0.5
        u = self.emb_dropout(u)

        u = u.unsqueeze(1)
        u = u.expand_as(seqs)

        cross = self.last_user_cross_layernorm6(self.dropout6(self.gating6(torch.cat((log_feats, log_feats2, att_category_features, att_category_features2),dim=2))))
        # cross = self.last_user_cross_layernorm3(self.dropout3(self.gating3(log_feats)))

        user = torch.cat((log_feats3, att_category_features3, cross, u), dim=2)
        user = self.last_user_cross_layernorm3(user)
        user = self.dropout3(self.gating3(user))  # 拼接

        user = self.lc_last_layernorm3(user)

        return user, con_loss1, con_loss2, con_loss3

    def get_user_metapath_embedding_UIU(self, user_ids):  # UIU:item与neighbors之间做attention
        users = self.user_emb(torch.LongTensor(user_ids).to(self.dev))  # 对序列做embedding  B,H
        users *= self.user_emb.embedding_dim ** 0.5  # /一个数，有点像attention点积时 /dim
        users = self.emb_dropout(users)

        neighbor_users = []
        for u in user_ids:
            neighbor = self.UIU_adj[u]
            while len(neighbor) < self.max_neighbor_len:
                neighbor.append(0)
            neighbor_users.append(neighbor)
        neighbor_users = torch.LongTensor(neighbor_users)  # B,K

        Q = self.uiu_attention_layernorms(users)  # B,H
        Q = Q.unsqueeze(1)

        K = self.user_emb(neighbor_users.to(self.dev))  # B,K,H
        # print(Q.shape,K.shape)
        K *= self.user_emb.embedding_dim ** 0.5
        K = V = self.emb_dropout(K)  # 得到一个B,C,H的矩阵,用来当K,V
        dot_products = torch.sum(Q * K, dim=-1)
        scaled_dot_products = dot_products / 50
        attention_weights = torch.softmax(scaled_dot_products, dim=-1)
        output = torch.sum(attention_weights.unsqueeze(-1) * V, dim=1)

        user_features = self.uiu_last_layernorm(output)
        return user_features

    def get_user_metapath_embedding_UICIU(self, user_ids):  # UIU:item与neighbors之间做attention
        users = self.user_emb(torch.LongTensor(user_ids).to(self.dev))  # 对序列做embedding  B,H
        users *= self.user_emb.embedding_dim ** 0.5  # /一个数，有点像attention点积时 /dim
        users = self.emb_dropout(users)

        neighbor_users = []
        for u in user_ids:
            neighbor = self.UICIU_adj[u]
            while len(neighbor) < self.max_neighbor_len:
                neighbor.append(0)
            neighbor_users.append(neighbor)
        neighbor_users = torch.LongTensor(neighbor_users)  # B,K

        Q = self.uiciu_attention_layernorms(users)  # B,H
        Q = Q.unsqueeze(1)

        K = self.user_emb(neighbor_users.to(self.dev))  # B,K,H
        # print(Q.shape,K.shape)
        K *= self.user_emb.embedding_dim ** 0.5
        K = V = self.emb_dropout(K)  # 得到一个B,C,H的矩阵,用来当K,V
        dot_products = torch.sum(Q * K, dim=-1)
        scaled_dot_products = dot_products / 50
        attention_weights = torch.softmax(scaled_dot_products, dim=-1)
        output = torch.sum(attention_weights.unsqueeze(-1) * V, dim=1)

        user_features = self.uiciu_last_layernorm(output)
        return user_features

    def get_category_metapath_embedding_CIC(self):  # CIC:category与neighbors之间做attention
        category_ids = list(range(0, self.category_num))    # 1,C
        categories = self.category_emb(torch.LongTensor(category_ids).reshape(-1, 1).to(self.dev))  # 对序列做embedding  C,H
        categories *= self.category_emb.embedding_dim ** 0.5  # /一个数，有点像attention点积时 /dim
        categories = self.emb_dropout(categories)

        neighbor_categories = []
        for c in category_ids:
            if c == 0:
                neighbor_categories.append([0]*self.max_neighbor_len)
                continue
            neighbor = self.CIC_adj[c]
            while len(neighbor) < self.max_neighbor_len:
                neighbor.append(0)
            neighbor_categories.append(neighbor)
        neighbor_categories = torch.LongTensor(neighbor_categories)  # C,K

        Q = self.cic_attention_layernorms(categories)  # C,1,H
        # Q = Q.unsqueeze(1)
        # print(Q.shape, neighbor_categories.shape)
        # import numpy as np
        # np.set_printoptions(threshold=np.inf)
        # print(np.array(neighbor_categories))

        K = self.category_emb(neighbor_categories.to(self.dev))  # C,K,H
        # print(Q.shape,K.shape)
        K *= self.category_emb.embedding_dim ** 0.5
        K = V = self.emb_dropout(K)  # 得到一个B,C,H的矩阵,用来当K,V
        dot_products = torch.sum(Q * K, dim=-1)
        scaled_dot_products = dot_products / 50
        attention_weights = torch.softmax(scaled_dot_products, dim=-1)
        # print(attention_weights.shape)
        output = torch.sum(attention_weights.unsqueeze(-1) * V, dim=1)

        category_features = self.cic_last_layernorm(output)  # C,H
        # print(category_features.shape)
        return category_features

    def get_category_metapath_embedding_CIUIC(self):  # UIU:item与neighbors之间做attention
        category_ids = list(range(0, self.category_num))
        categories = self.category_emb(torch.LongTensor(category_ids).reshape(-1, 1).to(self.dev))  # 对序列做embedding  B,H
        categories *= self.category_emb.embedding_dim ** 0.5  # /一个数，有点像attention点积时 /dim
        categories = self.emb_dropout(categories)

        neighbor_categories = []
        for c in category_ids:
            if c == 0:
                neighbor_categories.append([0]*self.max_neighbor_len)
                continue
            neighbor = self.CIUIC_adj[c]
            while len(neighbor) < self.max_neighbor_len:
                neighbor.append(0)
            neighbor_categories.append(neighbor)
        neighbor_categories = torch.LongTensor(neighbor_categories)  # B,K

        Q = self.ciuic_attention_layernorms(categories)  # B,H
        # Q = Q.unsqueeze(1)

        K = self.category_emb(neighbor_categories.to(self.dev))  # B,K,H
        # print(Q.shape,K.shape)
        K *= self.category_emb.embedding_dim ** 0.5
        K = V = self.emb_dropout(K)  # 得到一个B,C,H的矩阵,用来当K,V
        dot_products = torch.sum(Q * K, dim=-1)
        scaled_dot_products = dot_products / 50
        attention_weights = torch.softmax(scaled_dot_products, dim=-1)
        output = torch.sum(attention_weights.unsqueeze(-1) * V, dim=1)

        category_features = self.ciuic_last_layernorm(output)
        return category_features

    def get_item_metapath_embedding_IUI(self, log_seqs):  # IUI:item与neighbors之间做attention
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))  # 对序列做embedding  B,L,H
        seqs *= self.item_emb.embedding_dim ** 0.5  # /一个数，有点像attention点积时 /dim
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        neighbor_seqs = []
        for b in log_seqs:
            neighbor_seq = []
            for j in b:
                if j == 0:
                    neighbor_seq.append([0] * self.max_neighbor_len)
                else:
                    cd = self.IUI_adj[j]
                    # cd.append(0*(max_calen-len(cd)))
                    while len(cd) < self.max_neighbor_len:
                        cd.append(0)
                    neighbor_seq.append(cd)
            neighbor_seqs.append(neighbor_seq)
        neighbor_seqs = torch.LongTensor(neighbor_seqs)  # B,L,K
        # neighbor_seqs = np.zeros((len(log_seqs), 100, self.max_neighbor_len), dtype=np.long)
        # for i, b in enumerate(log_seqs):
        #     neighbor_seqs[i, :len(b), :] = np.array(
        #         [self.IUI_adj[j] + [0] * (self.max_neighbor_len - len(self.IUI_adj[j])) for j in b], dtype=np.long)
        # neighbor_seqs = torch.LongTensor(neighbor_seqs)

        Q = self.iui_attention_layernorms(seqs)
        Q = Q.unsqueeze(2)

        K = self.item_emb(neighbor_seqs.to(self.dev))  # B,L,C,H
        K *= self.item_emb.embedding_dim ** 0.5
        K = V = self.emb_dropout(K)  # 得到一个B,C,H的矩阵,用来当K,V
        dot_products = torch.sum(Q * K, dim=-1)
        scaled_dot_products = dot_products / 50
        attention_weights = torch.softmax(scaled_dot_products, dim=-1)
        output = torch.sum(attention_weights.unsqueeze(-1) * V, dim=2)

        item_features = self.iui_last_layernorm(output)
        return item_features

    def get_item_metapath_embedding_ICI(self, log_seqs):  # ICI:item与neighbors之间做attention
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))  # 对序列做embedding  B,L,H
        seqs *= self.item_emb.embedding_dim ** 0.5  # /一个数，有点像attention点积时 /dim
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        neighbor_seqs = []
        for b in log_seqs:
            neighbor_seq = []
            for j in b:
                if j == 0:
                    neighbor_seq.append([0] * self.max_neighbor_len)
                else:
                    cd = self.ICI_adj[j]
                    # cd.append(0*(max_calen-len(cd)))
                    while len(cd) < self.max_neighbor_len:
                        cd.append(0)
                    neighbor_seq.append(cd)
            neighbor_seqs.append(neighbor_seq)
        neighbor_seqs = torch.LongTensor(neighbor_seqs)  # B,L,K

        # neighbor_seqs = np.zeros((len(log_seqs), 100, self.max_neighbor_len), dtype=np.long)
        # for i, b in enumerate(log_seqs):
        #     neighbor_seqs[i, :len(b), :] = np.array(
        #         [self.ICI_adj[j] + [0] * (self.max_neighbor_len - len(self.ICI_adj[j])) for j in b], dtype=np.long)
        # neighbor_seqs = torch.LongTensor(neighbor_seqs)

        Q = self.ici_attention_layernorms(seqs)
        Q = Q.unsqueeze(2)

        K = self.item_emb(neighbor_seqs.to(self.dev))  # B,L,C,H
        K *= self.item_emb.embedding_dim ** 0.5
        K = V = self.emb_dropout(K)  # 得到一个B,C,H的矩阵,用来当K,V
        dot_products = torch.sum(Q * K, dim=-1)
        scaled_dot_products = dot_products / 50
        attention_weights = torch.softmax(scaled_dot_products, dim=-1)
        output = torch.sum(attention_weights.unsqueeze(-1) * V, dim=2)

        item_features = self.ici_last_layernorm(output)
        return item_features

    def get_item_category_att(self, log_seqs):  # item与category之间做attention
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))  # 对序列做embedding  B,L,H
        seqs *= self.item_emb.embedding_dim ** 0.5  # /一个数，有点像attention点积时 /dim
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        ca_seqs = []
        for b in log_seqs:
            ca_seq = []
            for j in b:
                cd = self.category[j]
                # cd.append(0*(max_calen-len(cd)))
                if len(cd) > self.max_category_len:
                    cd = cd[:self.max_category_len]

                while len(cd) < self.max_category_len:
                    cd.append(0)
                ca_seq.append(cd)
            ca_seqs.append(ca_seq)
        ca_seqs = torch.LongTensor(ca_seqs)  # B,L,C

        # print("1: \n",ca_seqs)
        # ca_seqs = np.zeros((len(log_seqs), 100, self.max_category_len), dtype=np.long)
        # for i, b in enumerate(log_seqs):
        #     ca_seqs[i, :len(b), :] = np.array(
        #         [self.category[j] + [0] * (self.max_category_len - len(self.category[j])) for j in b], dtype=np.long)
        # ca_seqs = torch.LongTensor(ca_seqs)

        # print("2:  \n", ca_seqs)
        # 转换为torch tensor
        # print(ca_seqs.shape)
        # ca_seqs = torch.transpose(ca_seqs, 0, 1)  # L,B,C
        #
        # seqs = torch.transpose(seqs, 0, 1)  # L,B,H

        # self.category_emb.weight = torch.nn.Parameter(self.GCN(self.category_adj, self.category_emb.weight))      # GCN

        # Q: (batch_size, num_queries, hidden_dim)
        # K: (batch_size, num_keys, channels, hidden_dim)
        # V: (batch_size, num_keys, channels, hidden_dim)
        # output: (batch_size, num_queries, hidden_dim)

        Q = self.ic_attention_layernorms(seqs)
        # Expand Q to match the shape of K
        Q = Q.unsqueeze(2)

        K = self.category_emb(ca_seqs.to(self.dev))  # B,L,C,H
        K *= self.category_emb.embedding_dim ** 0.5
        K = V = self.emb_dropout(K)  # 得到一个B,C,H的矩阵,用来当K,V
        #         print(K.shape, Q.shape)

        # Compute the dot product between Q and K
        dot_products = torch.sum(Q * K, dim=-1)

        # Scale the dot products by the square root of the key dimension
        scaled_dot_products = dot_products / 50

        # Compute the softmax over the last dimension of the scaled dot products
        attention_weights = torch.softmax(scaled_dot_products, dim=-1)

        # Compute the weighted sum of V using the attention weights
        output = torch.sum(attention_weights.unsqueeze(-1) * V, dim=2)

        # category_features_list = []
        # for c, s in zip(ca_seqs, seqs):
        #     c_emb = self.category_emb(c.to(self.dev))  # B,C,H
        #     # c_emb =
        #     c_emb *= self.category_emb.embedding_dim ** 0.5
        #     c_emb = self.emb_dropout(c_emb)  # 得到一个B,C,H的矩阵,用来当K,V
        #
        #     timeline_mask = torch.BoolTensor(c == 0).to(self.dev)
        #     c_emb *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim
        #
        #     s = s.unsqueeze(1)  # B,1,H
        #     Q = self.ic_attention_layernorms(s)
        #     mha_outputs, _ = self.ic_attention_layers(Q, c_emb, c_emb)
        #     # print(mha_outputs.shape)    # B,1,H
        #     # mha_outputs += c_emb
        #     # mha_outputs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim
        #     category_features_list.append(mha_outputs)
        #
        # category_features = torch.cat(category_features_list, dim=1).to(self.dev)
        # print(category_features.shape)    # B,L,H

        category_features = self.ic_last_layernorm(output)
        return category_features


    # def get_item_category_att(self, log_seqs):  # item与category之间做attention
    #     seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))  # 对序列做embedding  B,L,H
    #     seqs *= self.item_emb.embedding_dim ** 0.5  # /一个数，有点像attention点积时 /dim
    #     seqs = self.emb_dropout(seqs)
    #
    #     timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
    #     seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim
    #
    #     ca_seqs = []
    #     for b in log_seqs:
    #         ca_seq = []
    #         for j in b:
    #             cd = self.category[j]
    #             # cd.append(0*(max_calen-len(cd)))
    #             while len(cd) < self.max_category_len:
    #                 cd.append(0)
    #             ca_seq.append(cd)
    #         ca_seqs.append(ca_seq)
    #     ca_seqs = torch.LongTensor(ca_seqs)  # B,L,C
    #
    #     # ca_seqs = np.zeros((len(log_seqs), 100, self.max_category_len), dtype=np.long)
    #     # for i, b in enumerate(log_seqs):
    #     #     ca_seqs[i, :len(b), :] = np.array(
    #     #         [self.category[j] + [0] * (self.max_category_len - len(self.category[j])) for j in b], dtype=np.long)
    #
    #     # 转换为torch tensor
    #     # ca_seqs = torch.LongTensor(ca_seqs)
    #     # print(ca_seqs.shape)
    #     ca_seqs = torch.transpose(ca_seqs, 0, 1)  # L,B,C
    #
    #     seqs = torch.transpose(seqs, 0, 1)
    #
    #     # self.category_emb.weight = torch.nn.Parameter(self.GCN(self.category_adj, self.category_emb.weight))      # GCN
    #
    #     category_features_list = []
    #     for c, s in zip(ca_seqs, seqs):
    #         c_emb = self.category_emb(c.to(self.dev))  # B,C,H
    #         # c_emb =
    #         c_emb *= self.category_emb.embedding_dim ** 0.5
    #         c_emb = self.emb_dropout(c_emb)  # 得到一个B,C,H的矩阵,用来当K,V
    #
    #         timeline_mask = torch.BoolTensor(c == 0).to(self.dev)
    #         c_emb *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim
    #
    #         s = s.unsqueeze(1)  # B,1,H
    #         Q = self.ic_attention_layernorms(s)
    #         mha_outputs, _ = self.ic_attention_layers(Q, c_emb, c_emb)
    #         # print(mha_outputs.shape)    # B,1,H
    #         # mha_outputs += c_emb
    #         # mha_outputs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim
    #         category_features_list.append(mha_outputs)
    #
    #     category_features = torch.cat(category_features_list, dim=1).to(self.dev)
    #     # print(category_features.shape)    # B,L,H
    #
    #     category_features = self.ic_last_layernorm(category_features)
    #     return category_features

    def category_attention(self, log_seqs, category_features):  # category之间做attention
        # seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))  # 对序列做embedding  # batch, len, hidden_size
        category_features *= self.category_emb.embedding_dim ** 0.5  # /一个数，有点像attention点积时 /dim
        category_features = self.emb_dropout(category_features)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        category_features *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = category_features.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.ca_attention_layers)):  # 子模块的每一个网络 ?为什么不直接用
            category_features = torch.transpose(category_features, 0, 1)  # 转置，为了符合attention输入
            Q = self.ca_attention_layernorms[i](category_features)  # LN后的作为Q？ item seq
            mha_outputs, _ = self.ca_attention_layers[i](Q, category_features, category_features,
                                                         # attention Q,K,V   # 测试发现全设置为Q影响也不大
                                                         attn_mask=attention_mask)
            # key_padding_mask=timeline_mask
            # need_weights=False) this arg do not work?
            category_features = Q + mha_outputs  # 残差
            category_features = torch.transpose(category_features, 0, 1)  # 转回来

            category_features = self.ca_forward_layernorms[i](category_features)  # LN
            category_features = self.ca_forward_layers[i](category_features)  # 前馈层
            category_features *= ~timeline_mask.unsqueeze(-1)

        category_features = self.ca_last_layernorm(category_features)
        return category_features

    def forward(self, user_ids, log_seqs, log_seqs2, log_seqs3, pos_seqs, neg_seqs, mask1, mask2, target):  # for training
        if target == "A":
            log_feats, con_loss1, con_loss2, con_loss3 = self.log2feats(user_ids, log_seqs, log_seqs2, log_seqs3, mask1, mask2)  # user_ids hasn't been used yet    # t时刻的输出
        elif target == "B":
            log_feats, con_loss1, con_loss2, con_loss3 = self.log2feats2(user_ids, log_seqs, log_seqs2, log_seqs3, mask1, mask2)  # user_ids hasn't been used yet    # t时刻的输出
        else:
            log_feats, con_loss1, con_loss2, con_loss3 = self.log2feats3(user_ids, log_seqs, log_seqs2, log_seqs3, mask1, mask2)  # user_ids hasn't been used yet    # t时刻的输出


        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)  # 正样本r
        neg_logits = (log_feats * neg_embs).sum(dim=-1)  # 负样本r

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits, con_loss1 / (log_feats.shape[0] * log_feats.shape[1]), con_loss2 / self.category_num, con_loss3 / log_feats.shape[0]  # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, log_seqs2, log_seqs3, item_indices, mask1, mask2, target):  # for inference
        if target == "A":
            log_feats, con_loss1, con_loss2, con_loss3 = self.log2feats(user_ids, log_seqs, log_seqs2, log_seqs3, mask1, mask2)  # user_ids hasn't been used yet    # t时刻的输出
        elif target == "B":
            log_feats, con_loss1, con_loss2, con_loss3 = self.log2feats2(user_ids, log_seqs, log_seqs2, log_seqs3, mask1, mask2)  # user_ids hasn't been used yet    # t时刻的输出
        else:
            log_feats, con_loss1, con_loss2, con_loss3 = self.log2feats3(user_ids, log_seqs, log_seqs2, log_seqs3, mask1, mask2)  # user_ids hasn't been used yet    # t时刻的输出

        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # print(logits.shape) # 1,101

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits  # , pos_logits2, neg_logits2


def norm(adj):
    adj += np.eye(adj.shape[0])  # 为每个结点增加自环
    degree = np.array(adj.sum(1))  # 为每个结点计算度
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

def SSL(sess_emb_hgnn, sess_emb_lgcn):  # 一个负采样
    def row_shuffle(embedding):
        corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
        return corrupted_embedding

    def row_column_shuffle(embedding):
        corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
        corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
        return corrupted_embedding

    def score(x1, x2):
        return torch.sum(torch.mul(x1, x2), 2)

    # print(sess_emb_hgnn.shape)
    pos = score(sess_emb_hgnn, sess_emb_lgcn)
    # neg1 = score(sess_emb_lgcn, row_column_shuffle(sess_emb_hgnn))
    neg1 = score(sess_emb_hgnn, row_column_shuffle(sess_emb_lgcn))
    # print(pos.shape)
    # print(neg1.shape)
    # one = torch.cuda.FloatTensor((neg1.shape[0],neg1.shape[1])).fill_(1) #0?
    one = torch.ones((neg1.shape[0], neg1.shape[1])).cuda()
    # print(torch.sigmoid(pos).shape)
    # print(one.shape)
    # one = zeros = torch.ones(neg1.shape[0])
    con_loss = torch.sum(-torch.log(1e-7 + torch.sigmoid(pos)) - torch.log(1e-7 + (one - torch.sigmoid(neg1))))
    return con_loss

def SSL2(sess_emb_hgnn, sess_emb_lgcn):  # 一个负采样
    def row_column_shuffle(embedding):
        corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
        # corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
        return corrupted_embedding

    def score(x1, x2):
        return torch.sum(torch.mul(x1.detach(), x2.detach()), 1)

    pos = score(sess_emb_hgnn, sess_emb_lgcn)
    # print(pos.shape)
    neg1 = score(sess_emb_hgnn, row_column_shuffle(sess_emb_lgcn))
    # print(neg1.shape)
    one = torch.ones((neg1.shape[0])).cuda()
    # print(one.shape)

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
