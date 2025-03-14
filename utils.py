import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import re
from transformers import BertTokenizer, BertModel
from tqdm import tqdm


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, user_train2, user_train3, time1, time2, time3, usernum, itemnums, batch_size, maxlen,
                    result_queue, SEED, target_index):
    def sample():
        user = np.random.randint(1, usernum + 1)
        if target_index == 0:
            while len(user_train[user]) <= 1:
                user = np.random.randint(1, usernum + 1)
        elif target_index == 1:
            while len(user_train2[user]) <= 1:
                user = np.random.randint(1, usernum + 1)
        else:
            while len(user_train3[user]) <= 1:
                user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        seq2 = np.zeros([maxlen], dtype=np.int32)
        seq3 = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        t1 = np.zeros([maxlen], dtype=np.int32)
        t2 = np.zeros([maxlen], dtype=np.int32)
        t3 = np.zeros([maxlen], dtype=np.int32)
        if target_index == 0:
            nxt = user_train[user][-1]
            ts = set(user_train[user])
            main_seq = user_train[user][:-1]
            main_time = time1[user][:-1]
            item_range_start = 1
            item_range_end = itemnums[0] + 1
        elif target_index == 1:
            nxt = user_train2[user][-1]
            ts = set(user_train2[user])
            main_seq = user_train2[user][:-1]
            main_time = time2[user][:-1]
            item_range_start = itemnums[0] + 1
            item_range_end = itemnums[0] + itemnums[1] + 1
        else:
            nxt = user_train3[user][-1]
            ts = set(user_train3[user])
            main_seq = user_train3[user][:-1]
            main_time = time3[user][:-1]
            item_range_start = itemnums[0] + itemnums[1] + 1
            item_range_end = itemnums[0] + itemnums[1] + itemnums[2] + 1

        idx = maxlen - 1
        for i, t in reversed(list(zip(main_seq, main_time))):
            if target_index == 0:
                seq[idx] = i
                t1[idx] = t
            elif target_index == 1:
                seq2[idx] = i
                t2[idx] = t
            else:
                seq3[idx] = i
                t3[idx] = t
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(item_range_start, item_range_end, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        if target_index == 0:
            idx = maxlen - 1
            for i, t in reversed(list(zip(user_train2[user][:-1], time2[user][:-1]))):
                seq2[idx] = i
                t2[idx] = t
                idx -= 1
                if idx == -1:
                    break
            mask1 = np.zeros([maxlen], dtype=np.int32)
            idx2 = 0
            for idx in range(len(seq)):
                while idx2 < maxlen and t1[idx] >= t2[idx2]:
                    idx2 += 1
                mask1[idx] = idx2

            idx = maxlen - 1
            for i, t in reversed(list(zip(user_train3[user][:-1], time3[user][:-1]))):
                seq3[idx] = i
                t3[idx] = t
                idx -= 1
                if idx == -1:
                    break
            mask2 = np.zeros([maxlen], dtype=np.int32)
            idx3 = 0
            for idx in range(len(seq)):
                while idx3 < maxlen and t1[idx] >= t3[idx3]:
                    idx3 += 1
                mask2[idx] = idx3
        elif target_index == 1:
            idx = maxlen - 1
            for i, t in reversed(list(zip(user_train[user][:-1], time1[user][:-1]))):
                seq[idx] = i
                t1[idx] = t
                idx -= 1
                if idx == -1:
                    break

            idx = maxlen - 1
            for i, t in reversed(list(zip(user_train3[user][:-1], time3[user][:-1]))):
                seq3[idx] = i
                t3[idx] = t
                idx -= 1
                if idx == -1:
                    break

            mask1 = np.zeros([maxlen], dtype=np.int32)
            idx2 = 0
            for idx in range(len(seq2)):
                while idx2 < maxlen and t2[idx] >= t1[idx2]:
                    idx2 += 1
                mask1[idx] = idx2

            mask2 = np.zeros([maxlen], dtype=np.int32)
            idx3 = 0
            for idx in range(len(seq2)):
                while idx3 < maxlen and t2[idx] >= t3[idx3]:
                    idx3 += 1
                mask2[idx] = idx3
        else:
            idx = maxlen - 1
            for i, t in reversed(list(zip(user_train[user][:-1], time1[user][:-1]))):
                seq[idx] = i
                t1[idx] = t
                idx -= 1
                if idx == -1:
                    break

            idx = maxlen - 1
            for i, t in reversed(list(zip(user_train2[user][:-1], time2[user][:-1]))):
                seq2[idx] = i
                t2[idx] = t
                idx -= 1
                if idx == -1:
                    break

            mask1 = np.zeros([maxlen], dtype=np.int32)
            idx2 = 0
            for idx in range(len(seq3)):
                while idx2 < maxlen and t3[idx] >= t1[idx2]:
                    idx2 += 1
                mask1[idx] = idx2

            mask2 = np.zeros([maxlen], dtype=np.int32)
            idx3 = 0
            for idx in range(len(seq3)):
                while idx3 < maxlen and t3[idx] >= t2[idx3]:
                    idx3 += 1
                mask2[idx] = idx3

        return user, seq, pos, neg, seq2, mask1, seq3, mask2

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, User2, User3, time1, time2, time3, usernum, itemnums, batch_size=64, maxlen=10, n_workers=4,
                 target_index=0):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                       User2,
                                                       User3,
                                                       time1,
                                                       time2,
                                                       time3,
                                                       usernum,
                                                       itemnums,
                                                       batch_size,
                                                       maxlen,
                                                       self.result_queue,
                                                       np.random.randint(2e9),
                                                       target_index
                                                       )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def common_evaluate(model, dataset, args, target_index, is_valid):
    [train, valid, test, usernum, itemnum1, neg, user_train2, user_valid2, user_test2, itemnum2, neg2, user_train3,
     user_valid3, user_test3, itemnum3, neg3, time1, time2, time3, category, category_emb, User_Item, Item_User,
     category_contain, category_num] = copy.deepcopy(dataset)

    NDCG_5 = 0.0
    HT_5 = 0.0
    NDCG_10 = 0.0
    HT_10 = 0.0
    NDCG_20 = 0.0
    HT_20 = 0.0
    valid_user = 0.0

    users = range(1, usernum + 1)
    for u in tqdm(users, desc="Evaluating", ncols=80):
    # for u in users:
        if target_index == 0:
            if (len(train[u]) < 1 and len(valid[u]) < 1 and is_valid) or (len(train[u]) < 1 and len(test[u]) < 1 and not is_valid):
                continue
            main_train = train[u]
            main_valid = valid[u]
            main_test = test[u]
            main_neg = neg[u]
            main_time = time1[u]
            other_train2 = user_train2[u]
            other_time2 = time2[u]
            other_train3 = user_train3[u]
            other_time3 = time3[u]
        elif target_index == 1:
            if (len(user_train2[u]) < 1 and len(user_valid2[u]) < 1 and is_valid) or (len(user_train2[u]) < 1 and len(user_test2[u]) < 1 and not is_valid):
                continue
            main_train = user_train2[u]
            main_valid = user_valid2[u]
            main_test = user_test2[u]
            main_neg = neg2[u]
            main_time = time2[u]
            other_train2 = train[u]
            other_time2 = time1[u]
            other_train3 = user_train3[u]
            other_time3 = time3[u]
        else:
            if (len(user_train3[u]) < 1 and len(user_valid3[u]) < 1 and is_valid) or (len(user_train3[u]) < 1 and len(user_test3[u]) < 1 and not is_valid):
                continue
            main_train = user_train3[u]
            main_valid = user_valid3[u]
            main_test = user_test3[u]
            main_neg = neg3[u]
            main_time = time3[u]
            other_train2 = train[u]
            other_time2 = time1[u]
            other_train3 = user_train2[u]
            other_time3 = time2[u]

        seq = np.zeros([args.maxlen], dtype=np.int32)
        t1 = np.zeros([args.maxlen], dtype=np.int32)
        t2 = np.zeros([args.maxlen], dtype=np.int32)
        t3 = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        if not is_valid:
            seq[idx] = main_valid[0]
            idx -= 1
        for i, t in reversed(list(zip(main_train, main_time))):
            seq[idx] = i
            t1[idx] = t
            idx -= 1
            if idx == -1:
                break
        rated = set(main_train)
        rated.add(0)
        item_idx = [main_test[0] if not is_valid else main_valid[0]]

        seq2 = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i, t in reversed(list(zip(other_train2, other_time2))):
            seq2[idx] = i
            t2[idx] = t
            idx -= 1
            if idx == -1:
                break
        mask2 = np.zeros([args.maxlen], dtype=np.int32)
        idx2 = 0
        for idx in range(len(seq)):
            while idx2 < args.maxlen and t1[idx] >= t2[idx2]:
                idx2 += 1
            mask2[idx] = idx2

        seq3 = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i, t in reversed(list(zip(other_train3, other_time3))):
            seq3[idx] = i
            t3[idx] = t
            idx -= 1
            if idx == -1:
                break
        mask3 = np.zeros([args.maxlen], dtype=np.int32)
        idx3 = 0
        for idx in range(len(seq)):
            while idx3 < args.maxlen and t1[idx] >= t3[idx3]:
                idx3 += 1
            mask3[idx] = idx3

        for i in main_neg:
            item_idx.append(i)

        if target_index == 0:
            predictions = -model.predict(
                *[np.array(l) for l in [[u], [seq], [seq2], [seq3], item_idx, [mask2], [mask3], "A"]])
        elif target_index == 1:
            predictions = -model.predict(
                *[np.array(l) for l in [[u], [seq2], [seq], [seq3], item_idx, [mask2], [mask3], "B"]])
        elif target_index == 2:
            predictions = -model.predict(
                *[np.array(l) for l in [[u], [seq2], [seq3], [seq], item_idx, [mask2], [mask3], "C"]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 5:
            NDCG_5 += 1 / np.log2(rank + 2)
            HT_5 += 1
        if rank < 10:
            NDCG_10 += 1 / np.log2(rank + 2)
            HT_10 += 1
        if rank < 20:
            NDCG_20 += 1 / np.log2(rank + 2)
            HT_20 += 1

    return NDCG_5 / valid_user, HT_5 / valid_user, NDCG_10 / valid_user, HT_10 / valid_user, NDCG_20 / valid_user, HT_20 / valid_user


def evaluate(model, dataset, args):
    return common_evaluate(model, dataset, args, 0, is_valid=False)


def evaluate_valid(model, dataset, args):
    return common_evaluate(model, dataset, args, 0, is_valid=True)

def evaluate2(model, dataset, args):
    return common_evaluate(model, dataset, args, 1, is_valid=False)


def evaluate_valid2(model, dataset, args):
    return common_evaluate(model, dataset, args, 1, is_valid=True)


def evaluate3(model, dataset, args):
    return common_evaluate(model, dataset, args, 2, is_valid=False)


def evaluate_valid3(model, dataset, args):
    return common_evaluate(model, dataset, args, 2, is_valid=True)


# train/val/test data generation
s = re.compile(r"['\"\[\]]")
category_num = 1
category2id = {'NULL': 0}


def read_data(fname, user_ids, item_ids, User, Time, Category):
    global category_num
    for phase in ['train', 'valid', 'test']:
        with open(f'cross_data/processed_data_all/{fname}_{phase}.csv', 'r') as f:
            for line in f:
                u, i, t, c = line.rstrip().split(',', 3)
                u = int(u)
                i = int(i)
                t = int(t)
                c = re.sub(s, "", c).split(', ')
                user_ids.append(u)
                item_ids.append(i)
                User[u].append(i)
                Time[u].append(t)
                cid = []
                if c[0] == '':
                    cid.append(0)
                else:
                    for cc in c[1:]:
                        if cc not in category2id:
                            category2id[cc] = category_num
                            category_num += 1
                        cid.append(category2id[cc])
                Category[i] = cid
    return user_ids, item_ids, User, Time, Category


def read_negative_samples(fname, user_map, item_map, neglist):
    with open(f'cross_data/processed_data_all/{fname}_negative.csv', 'r') as f:
        for line in f:
            l = line.rstrip().split(',')
            u = user_map[int(l[0])]
            for j in range(1, 101):
                i = item_map[int(l[j])]
                neglist[u].append(i)
    return neglist


def split_data(User, user_train, user_valid, user_test, user_neg, neglist):
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
        user_neg[user] = neglist[user]
    return user_train, user_valid, user_test, user_neg


def data_partition(fname, fname2, fname3):
    usernum = 0
    itemnum1 = 0
    User = defaultdict(list)
    User1 = defaultdict(list)
    User2 = defaultdict(list)
    User3 = defaultdict(list)
    user_train1 = {}
    user_valid1 = {}
    user_test1 = {}
    neglist1 = defaultdict(list)
    neglist2 = defaultdict(list)
    neglist3 = defaultdict(list)
    user_neg1 = {}
    user_neg2 = {}
    user_neg3 = {}

    user_map = dict()
    item_map = dict()

    user_ids = list()
    item_ids1 = list()

    itemnum2 = 0
    user_train2 = {}
    user_valid2 = {}
    user_test2 = {}
    item_ids2 = list()

    itemnum3 = 0
    user_train3 = {}
    user_valid3 = {}
    user_test3 = {}
    item_ids3 = list()

    Time = defaultdict(list)
    Time1 = {}
    Time2 = {}
    Time3 = {}

    Category = dict()
    category = {}

    user_ids, item_ids1, User, Time, Category = read_data(fname, user_ids, item_ids1, User, Time, Category)
    for u in user_ids:
        if u not in user_map:
            user_map[u] = usernum + 1
            usernum += 1
    for i in item_ids1:
        if i not in item_map:
            item_map[i] = itemnum1 + 1
            itemnum1 += 1

    neglist1 = read_negative_samples(fname, user_map, item_map, neglist1)

    for user in User:
        u = user_map[user]
        for item in User[user]:
            i = item_map[item]
            User1[u].append(i)
        Time1[u] = Time[user]

    User = defaultdict(list)
    Time = defaultdict(list)

    user_ids, item_ids2, User, Time, Category = read_data(fname2, user_ids, item_ids2, User, Time, Category)
    for i in item_ids2:
        if i not in item_map:
            item_map[i] = itemnum1 + itemnum2 + 1
            itemnum2 += 1

    for user in User:
        u = user_map[user]
        for item in User[user]:
            i = item_map[item]
            User2[u].append(i)
        Time2[u] = Time[user]

    User = defaultdict(list)
    Time = defaultdict(list)

    user_ids, item_ids3, User, Time, Category = read_data(fname3, user_ids, item_ids3, User, Time, Category)
    for i in item_ids3:
        if i not in item_map:
            item_map[i] = itemnum1 + itemnum2 + itemnum3 + 1
            itemnum3 += 1

    for user in User:
        u = user_map[user]
        for item in User[user]:
            i = item_map[item]
            User3[u].append(i)
        Time3[u] = Time[user]

    for i in Category.keys():
        item = item_map[i]
        category[item] = Category[i]
    category[0] = [0]

    print("category_num: ", category_num)
    print("itemnum1: ", itemnum1)
    print("itemnum2: ", itemnum2)
    print("itemnum3: ", itemnum3)

    model_path = 'pretrained/'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)

    words = []
    for key in category2id.keys():
        words.append(key)

    embeddings = []
    for word in tqdm(words, desc="Processing Word Embedding", ncols=80):
        subwords = tokenizer.tokenize(word)
        input_ids = tokenizer.convert_tokens_to_ids(subwords)
        input_tensor = torch.tensor([input_ids])

        with torch.no_grad():
            output_tensor = model(input_tensor)

        last_hidden_state = output_tensor.last_hidden_state
        last_hidden_state_for_word = last_hidden_state[0][-len(subwords):]
        embedding = torch.mean(last_hidden_state_for_word, dim=0).numpy()
        embeddings.append(embedding)

    category_emb = torch.Tensor(embeddings)

    data = np.array(embeddings)
    data = data / np.linalg.norm(data, axis=1, keepdims=True)

    similarity_ex = np.zeros((category_num, category_num))
    for i in range(1, data.shape[0]):
        for j in range(1, data.shape[0]):
            if i != j:
                similarity_ex[i][j] = np.dot(data[i], data[j]) / (np.linalg.norm(data[i]) * np.linalg.norm(data[j]))

    category_ex = get_category_adj(category, category2id, similarity_ex)

    for i in category.keys():
        if category[i][0] == 0:
            continue
        category_set = set()
        for c in category[i]:
            if c in category_ex.keys():
                for c_ex in category_ex[c]:
                    category_set.add(c_ex)
        category[i].extend(list(category_set))
        category[i] = list(set(category[i]))

    User_Item = {}
    Item_User = {}
    category_contain = {}

    for user, items in User1.items():
        User_Item[user] = set(items)
        for item in items:
            if item in Item_User:
                Item_User[item].add(user)
            else:
                Item_User[item] = {user}

    for user, items in User2.items():
        for item in items:
            User_Item[user].add(item)

            if item in Item_User:
                Item_User[item].add(user)
            else:
                Item_User[item] = {user}

    for user, items in User3.items():
        for item in items:
            User_Item[user].add(item)

            if item in Item_User:
                Item_User[item].add(user)
            else:
                Item_User[item] = {user}

    for item, categories in category.items():
        for cat in categories:
            if cat in category_contain:
                category_contain[cat].add(item)
            else:
                category_contain[cat] = {item}

    neglist2 = read_negative_samples(fname2, user_map, item_map, neglist2)
    neglist3 = read_negative_samples(fname3, user_map, item_map, neglist3)

    user_train1, user_valid1, user_test1, user_neg1 = split_data(User1, user_train1, user_valid1, user_test1, user_neg1, neglist1)
    user_train2, user_valid2, user_test2, user_neg2 = split_data(User2, user_train2, user_valid2, user_test2, user_neg2, neglist2)
    user_train3, user_valid3, user_test3, user_neg3 = split_data(User3, user_train3, user_valid3, user_test3, user_neg3, neglist3)

    return [user_train1, user_valid1, user_test1, usernum, itemnum1, user_neg1, user_train2, user_valid2, user_test2,
            itemnum2, user_neg2, user_train3, user_valid3, user_test3, itemnum3, user_neg3, Time1, Time2, Time3,
            category, category_emb, User_Item, Item_User,
            category_contain, category_num]


def get_category_adj(category, category2id, similarity_ex):
    print("Constructing Graph...")
    category_num = len(category2id)

    N = np.zeros(category_num)
    M = np.zeros((category_num, category_num))

    for values in category.values():
        for idx, v in enumerate(values):
            if v == 0:
                continue
            N[v] += 1
            for j in range(idx + 1, len(values)):
                if values[j] != 0:
                    M[v][values[j]] += 1
                    M[values[j]][v] += 1

    category_ex = {}
    for i in range(1, category_num):
        for j in range(1, category_num):
            p = M[i][j] / N[i]
            if p >= 0.5 or similarity_ex[i][j] >= 0.75:
                if i in category_ex:
                    category_ex[i].append(j)
                else:
                    category_ex[i] = [j]
    return category_ex
