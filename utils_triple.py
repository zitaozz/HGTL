import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import re
from transformers import BertTokenizer, BertModel


# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function1(user_train, user_train2, user_train3, time1, time2, time3, usernum, itemnum1, itemnum2, itemnum3, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        seq2 = np.zeros([maxlen], dtype=np.int32)
        seq3 = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        t1 = np.zeros([maxlen], dtype=np.int32)  #
        t2 = np.zeros([maxlen], dtype=np.int32)  #
        t3 = np.zeros([maxlen], dtype=np.int32)  #
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i, t in reversed(list(zip(user_train[user][:-1], time1[user][:-1]))):
            seq[idx] = i
            t1[idx] = t
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum1 + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        idx = maxlen - 1
        for i, t in reversed(list(zip(user_train2[user][:-1], time2[user][:-1]))):
            seq2[idx] = i
            t2[idx] = t
            idx -= 1
            if idx == -1: break

        mask2 = np.zeros([maxlen], dtype=np.int32)  #
        idx2 = 0
        for idx in range(len(seq)):
            while idx2 < maxlen and t1[idx] >= t2[idx2]:
                idx2 += 1
            mask2[idx] = idx2


        idx = maxlen - 1
        for i, t in reversed(list(zip(user_train3[user][:-1], time3[user][:-1]))):
            seq3[idx] = i
            t3[idx] = t
            idx -= 1
            if idx == -1: break

        mask3 = np.zeros([maxlen], dtype=np.int32)  #
        idx3 = 0
        for idx in range(len(seq)):
            while idx3 < maxlen and t1[idx] >= t3[idx3]:
                idx3 += 1
            mask3[idx] = idx3

        return (user, seq, pos, neg, seq2, mask2, seq3, mask3)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler1(object):
    def __init__(self, User, User2, User3, time1, time2, time3, usernum, itemnum1, itemnum2, itemnum3, batch_size=64, maxlen=10, n_workers=4):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function1, args=(User,
                                                      User2,
                                                      User3,
                                                      time1,
                                                      time2,
                                                      time3,
                                                      usernum,
                                                      itemnum1,
                                                      itemnum2,
                                                      itemnum3,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

def sample_function2(user_train, user_train2, user_train3, time1, time2, time3, usernum, itemnum1, itemnum2, itemnum3, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train2[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        seq2 = np.zeros([maxlen], dtype=np.int32)
        seq3 = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        t1 = np.zeros([maxlen], dtype=np.int32)  #
        t2 = np.zeros([maxlen], dtype=np.int32)  #
        t3 = np.zeros([maxlen], dtype=np.int32)  #
        nxt = user_train2[user][-1]
        idx = maxlen - 1

        ts = set(user_train2[user])
        for i, t in reversed(list(zip(user_train2[user][:-1], time2[user][:-1]))):
            seq2[idx] = i
            t2[idx] = t
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(itemnum1 + 1, itemnum1 + itemnum2 + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        idx = maxlen - 1
        for i, t in reversed(list(zip(user_train[user][:-1], time1[user][:-1]))):
            seq[idx] = i
            t1[idx] = t
            idx -= 1
            if idx == -1: break

        idx = maxlen - 1
        for i, t in reversed(list(zip(user_train3[user][:-1], time3[user][:-1]))):
            seq3[idx] = i
            t3[idx] = t
            idx -= 1
            if idx == -1: break

        mask1 = np.zeros([maxlen], dtype=np.int32)  #
        idx2 = 0
        for idx in range(len(seq2)):
            while idx2 < maxlen and t2[idx] >= t1[idx2]:
                idx2 += 1
            mask1[idx] = idx2

        mask3 = np.zeros([maxlen], dtype=np.int32)  #
        idx3 = 0
        for idx in range(len(seq2)):
            while idx3 < maxlen and t2[idx] >= t3[idx3]:
                idx3 += 1
            mask3[idx] = idx3

        return (user, seq, pos, neg, seq2, mask1, seq3, mask3)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler2(object):
    def __init__(self, User, User2, User3, time1, time2, time3, usernum, itemnum1, itemnum2, itemnum3, batch_size=64, maxlen=10, n_workers=4):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function2, args=(User,
                                                      User2,
                                                      User3,
                                                      time1,
                                                      time2,
                                                      time3,
                                                      usernum,
                                                      itemnum1,
                                                      itemnum2,
                                                      itemnum3,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

def sample_function3(user_train, user_train2, user_train3, time1, time2, time3, usernum, itemnum1, itemnum2, itemnum3, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train3[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        seq2 = np.zeros([maxlen], dtype=np.int32)
        seq3 = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        t1 = np.zeros([maxlen], dtype=np.int32)  #
        t2 = np.zeros([maxlen], dtype=np.int32)  #
        t3 = np.zeros([maxlen], dtype=np.int32)  #
        nxt = user_train3[user][-1]
        idx = maxlen - 1

        ts = set(user_train3[user])
        for i, t in reversed(list(zip(user_train3[user][:-1], time3[user][:-1]))):
            seq3[idx] = i
            t3[idx] = t
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(itemnum1 + itemnum2 + 1, itemnum1 + itemnum2 + itemnum3 + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        idx = maxlen - 1
        for i, t in reversed(list(zip(user_train[user][:-1], time1[user][:-1]))):
            seq[idx] = i
            t1[idx] = t
            idx -= 1
            if idx == -1: break

        idx = maxlen - 1
        for i, t in reversed(list(zip(user_train2[user][:-1], time2[user][:-1]))):
            seq2[idx] = i
            t2[idx] = t
            idx -= 1
            if idx == -1: break

        mask1 = np.zeros([maxlen], dtype=np.int32)  #
        idx2 = 0
        for idx in range(len(seq3)):
            while idx2 < maxlen and t3[idx] >= t1[idx2]:
                idx2 += 1
            mask1[idx] = idx2

        mask2 = np.zeros([maxlen], dtype=np.int32)  #
        idx3 = 0
        for idx in range(len(seq3)):
            while idx3 < maxlen and t3[idx] >= t2[idx3]:
                idx3 += 1
            mask2[idx] = idx3

        return (user, seq, pos, neg, seq2, mask1, seq3, mask2)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler3(object):
    def __init__(self, User, User2, User3, time1, time2, time3, usernum, itemnum1, itemnum2, itemnum3, batch_size=64, maxlen=10, n_workers=4):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function3, args=(User,
                                                      User2,
                                                      User3,
                                                      time1,
                                                      time2,
                                                      time3,
                                                      usernum,
                                                      itemnum1,
                                                      itemnum2,
                                                      itemnum3,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

# train/val/test data generation
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
    # Category[0] = [0]
    category = {}
    category2id = dict()
    category_num = 1
    category2id['NULL'] = 0

    s = re.compile(r"['\"\[\]]")
    # assume user/item index starting from 1
    f = open('cross_data/processed_data_all/%s_train.csv' % fname, 'r')
    for line in f:
        u, i, t, c = line.rstrip().split(',', 3)
        u = int(u)
        i = int(i)
        t = int(t)
        c = re.sub(s, "", c).split(', ')
        # usernum = max(u, usernum)
        # itemnum = max(i, itemnum)
        user_ids.append(u)
        item_ids1.append(i)
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

    f = open('cross_data/processed_data_all/%s_valid.csv' % fname, 'r')
    for line in f:
        u, i, t, c = line.rstrip().split(',', 3)
        u = int(u)
        i = int(i)
        t = int(t)
        # usernum = max(u, usernum)
        # itemnum = max(i, itemnum)
        c = re.sub(s, "", c).split(', ')
        user_ids.append(u)
        item_ids1.append(i)
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

    f = open('cross_data/processed_data_all/%s_test.csv' % fname, 'r')
    for line in f:
        u, i, t, c = line.rstrip().split(',', 3)
        u = int(u)
        i = int(i)
        t = int(t)
        # usernum = max(u, usernum)
        # itemnum = max(i, itemnum)
        c = re.sub(s, "", c).split(', ')
        user_ids.append(u)
        item_ids1.append(i)
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

    # update user and item mapping
    for u in user_ids:
        if u not in user_map:
            user_map[u] = usernum + 1
            usernum += 1
    for i in item_ids1:
        if i not in item_map:
            item_map[i] = itemnum1 + 1
            itemnum1 += 1

    # for i in Category.keys():
    #     item = item_map[i]
    #     category[item] = Category[i]
    # category[0] = [0]

    # print(max(category, key=lambda k:len(category[k])))  # 最长为11
    # s = 0
    # for key in category.keys():
    #     s += len(category[key])
    # print(s/len(category))     # 平均3

    f = open('cross_data/processed_data_all/%s_negative.csv' % fname, 'r')
    for line in f:
        l = line.rstrip().split(',')
        u = user_map[int(l[0])]
        for j in range(1, 101):
            i = item_map[int(l[j])]
            neglist1[u].append(i)

    # for item in item_map

    for user in User:
        u = user_map[user]
        for item in User[user]:
            i = item_map[item]
            User1[u].append(i)
        Time1[u] = Time[user]

    User = defaultdict(list)
    Time = defaultdict(list)

    f = open('cross_data/processed_data_all/%s_train.csv' % fname2, 'r')
    for line in f:
        u, i, t, c = line.rstrip().split(',', 3)
        u = int(u)
        i = int(i)
        t = int(t)
        c = re.sub(s, "", c).split(', ')
        # usernum = max(u, usernum)
        # itemnum = max(i, itemnum)
        user_ids.append(u)
        item_ids2.append(i)
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

    f = open('cross_data/processed_data_all/%s_valid.csv' % fname2, 'r')
    for line in f:
        u, i, t, c = line.rstrip().split(',', 3)
        u = int(u)
        i = int(i)
        t = int(t)
        # usernum = max(u, usernum)
        # itemnum = max(i, itemnum)
        c = re.sub(s, "", c).split(', ')
        user_ids.append(u)
        item_ids2.append(i)
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

    f = open('cross_data/processed_data_all/%s_test.csv' % fname2, 'r')
    for line in f:
        u, i, t, c = line.rstrip().split(',', 3)
        u = int(u)
        i = int(i)
        t = int(t)
        # usernum = max(u, usernum)
        # itemnum = max(i, itemnum)
        c = re.sub(s, "", c).split(', ')
        user_ids.append(u)
        item_ids2.append(i)
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

    # update user and item mapping
    # for u in user_ids:
    #     if u not in user_map:
    #         user_map[u] = usernum + 1
    #         usernum += 1
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

    f = open('cross_data/processed_data_all/%s_train.csv' % fname3, 'r')
    for line in f:
        u, i, t, c = line.rstrip().split(',', 3)
        u = int(u)
        i = int(i)
        t = int(t)
        c = re.sub(s, "", c).split(', ')
        # usernum = max(u, usernum)
        # itemnum = max(i, itemnum)
        user_ids.append(u)
        item_ids3.append(i)
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

    f = open('cross_data/processed_data_all/%s_valid.csv' % fname3, 'r')
    for line in f:
        u, i, t, c = line.rstrip().split(',', 3)
        u = int(u)
        i = int(i)
        t = int(t)
        # usernum = max(u, usernum)
        # itemnum = max(i, itemnum)
        c = re.sub(s, "", c).split(', ')
        user_ids.append(u)
        item_ids3.append(i)
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

    f = open('cross_data/processed_data_all/%s_test.csv' % fname3, 'r')
    for line in f:
        u, i, t, c = line.rstrip().split(',', 3)
        u = int(u)
        i = int(i)
        t = int(t)
        # usernum = max(u, usernum)
        # itemnum = max(i, itemnum)
        c = re.sub(s, "", c).split(', ')
        user_ids.append(u)
        item_ids3.append(i)
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

    # update user and item mapping
    # for u in user_ids:
    #     if u not in user_map:
    #         user_map[u] = usernum + 1
    #         usernum += 1
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

    # category_emb = dict()
    # 加载本地预训练的BERT模型和分词器
    model_path = 'pretrained/'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)

    # 要编码的单词列表
    words = []
    # words = ['apple', 'banana', 'cherry 123']
    for key in category2id.keys():
        words.append(key)
    #     print(words)

    # 对每个单词进行编码
    embeddings = []
    # embeddings.append([0]*768)
    for word in words:
        # 使用分词器将单词分成子词
        subwords = tokenizer.tokenize(word)

        # 将子词转换为模型的输入张量格式
        input_ids = tokenizer.convert_tokens_to_ids(subwords)
        input_tensor = torch.tensor([input_ids])

        # 使用模型对单词进行编码
        with torch.no_grad():
            output_tensor = model(input_tensor)

        # 取出单词的最后一层隐藏状态，并将编码向量添加到列表中
        last_hidden_state = output_tensor.last_hidden_state
        last_hidden_state_for_word = last_hidden_state[0][-len(subwords):]
        embedding = torch.mean(last_hidden_state_for_word, dim=0).numpy()
        embeddings.append(embedding)

    category_emb = torch.Tensor(embeddings)
    # 输出编码向量列表
    # print(torch.Tensor(embeddings).shape)   # C, 768

    # 将列表转换成numpy数组，并对每一行向量进行标准化处理
    data = np.array(embeddings)
    data = data / np.linalg.norm(data, axis=1, keepdims=True)

    # 计算每一行向量与其他行向量的余弦相似度，并记录大于0.8的下标
    # similarity_ex = {}
    similarity_ex = np.zeros((category_num, category_num))
    for i in range(1, data.shape[0]):
        for j in range(1, data.shape[0]):
            if i != j:
                similarity_ex[i][j] = np.dot(data[i], data[j]) / (np.linalg.norm(data[i]) * np.linalg.norm(data[j]))
            # similarity = np.dot(data[i], data[j]) / (np.linalg.norm(data[i]) * np.linalg.norm(data[j]))
            # if similarity > 0.8:
            #     if i in similarity_ex:
            #         similarity_ex[i].append(j)
            #     else:
            #         similarity_ex[i] = [j]

    category_ex = get_category_adj(category, category2id, similarity_ex)
    #
    # print("exc: ", len(category_ex))
    # for k, v in category_ex.items():
    #     s = next((key for key, value in category2id.items() if value == k), None)
    #     print(s, end=": ")
    #     for c in v:
    #         cc = next((key for key, value in category2id.items() if value == c), None)
    #         print(cc, end=", ")
    #     print()

    # suml = 0
    # maxl = 0
    for i in category.keys():
        if category[i][0] == 0:
            continue
        category_set = set()
        for c in category[i]:
            if c in category_ex.keys():
                # print(i, c, category[i], category_ex[c])
                # category[i].extend(category_ex[c])
                for c_ex in category_ex[c]:
                    category_set.add(c_ex)
            # if c in similarity_ex.keys():
            #     category[i].extend(similarity_ex[c])
        category[i].extend(list(category_set))
        category[i] = list(set(category[i]))
    # #     l = len(category[i])
    #     # print(l)
    #     maxl = max(l, maxl)
    #     suml += l
    # print(suml/len(category))
    # print(maxl)
    # for k,v in category.items():
    #     print(k, ": ", v)

    # avg category len = 10, max = 68 // 0.8: 3,20

    # 构造相关字典
    User_Item = {}  # 用户交互过的物品
    Item_User = {}  # 交互过物品的用户
    category_contain = {}  # 类别下的所有物品

    for user, items in User1.items():
        User_Item[user] = set(items)
        for item in items:
            if item in Item_User:
                Item_User[item].add(user)
            else:
                Item_User[item] = {user}

    for user, items in User2.items():
        # User_Item[user].add(set(items))
        for item in items:
            User_Item[user].add(item)

            if item in Item_User:
                Item_User[item].add(user)
            else:
                Item_User[item] = {user}

    for user, items in User3.items():
        # User_Item[user].add(set(items))
        for item in items:
            User_Item[user].add(item)

            if item in Item_User:
                Item_User[item].add(user)
            else:
                Item_User[item] = {user}

    # 构建category_contain字典
    for item, categories in category.items():
        for cat in categories:
            if cat in category_contain:
                category_contain[cat].add(item)
            else:
                category_contain[cat] = {item}

    # for k,v in category_contain.items():
    #     print(k, ": ", len(v))

    f = open('cross_data/processed_data_all/%s_negative.csv' % fname2, 'r')
    for line in f:
        l = line.rstrip().split(',')
        u = user_map[int(l[0])]
        for j in range(1, 101):
            i = item_map[int(l[j])]
            neglist2[u].append(i)

    f = open('cross_data/processed_data_all/%s_negative.csv' % fname3, 'r')
    for line in f:
        l = line.rstrip().split(',')
        u = user_map[int(l[0])]
        for j in range(1, 101):
            i = item_map[int(l[j])]
            neglist3[u].append(i)

    for user in User1:
        nfeedback = len(User1[user])
        if nfeedback < 3:
            user_train1[user] = User1[user]
            user_valid1[user] = []
            user_test1[user] = []
        else:
            user_train1[user] = User1[user][:-2]
            user_valid1[user] = []
            user_valid1[user].append(User1[user][-2])
            user_test1[user] = []
            user_test1[user].append(User1[user][-1])
        user_neg1[user] = neglist1[user]

    for user in User2:
        nfeedback = len(User2[user])
        if nfeedback < 3:
            user_train2[user] = User2[user]
            user_valid2[user] = []
            user_test2[user] = []
        else:
            user_train2[user] = User2[user][:-2]
            user_valid2[user] = []
            user_valid2[user].append(User2[user][-2])
            user_test2[user] = []
            user_test2[user].append(User2[user][-1])
        user_neg2[user] = neglist2[user]

    for user in User3:
        nfeedback = len(User3[user])
        if nfeedback < 3:
            user_train3[user] = User3[user]
            user_valid3[user] = []
            user_test3[user] = []
        else:
            user_train3[user] = User3[user][:-2]
            user_valid3[user] = []
            user_valid3[user].append(User3[user][-2])
            user_test3[user] = []
            user_test3[user].append(User3[user][-1])
        user_neg3[user] = neglist3[user]

    return [user_train1, user_valid1, user_test1, usernum, itemnum1, user_neg1, user_train2, user_valid2, user_test2, itemnum2, user_neg2, user_train3, user_valid3, user_test3, itemnum3, user_neg3, Time1, Time2, Time3, category, category_emb, User_Item, Item_User,
            category_contain]


# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum1, neg, user_train2, user_valid2, user_test2, itemnum2, neg2, user_train3, user_valid3, user_test3, itemnum3, neg3, time1, time2, time3, category, category_emb, User_Item, Item_User,
     category_contain] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    # if usernum>10000:
    #    users = random.sample(range(1, usernum + 1), 10000)
    # else:
    users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        t1 = np.zeros([args.maxlen], dtype=np.int32)  #
        t2 = np.zeros([args.maxlen], dtype=np.int32)  #
        t3 = np.zeros([args.maxlen], dtype=np.int32)  #
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i, t in reversed(list(zip(train[u], time1[u]))):
            seq[idx] = i
            t1[idx] = t
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]

        # for _ in range(100):
        #     t = np.random.randint(1, itemnum + 1)
        #     while t in rated: t = np.random.randint(1, itemnum + 1)
        #     item_idx.append(t)

        seq2 = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i, t in reversed(list(zip(user_train2[u], time2[u]))):
            seq2[idx] = i
            t2[idx] = t
            idx -= 1
            if idx == -1: break

        mask2 = np.zeros([args.maxlen], dtype=np.int32)  #
        idx2 = 0
        for idx in range(len(seq)):
            while idx2 < args.maxlen and t1[idx] >= t2[idx2]:
                idx2 += 1
            mask2[idx] = idx2

        seq3 = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i, t in reversed(list(zip(user_train3[u], time3[u]))):
            seq3[idx] = i
            t3[idx] = t
            idx -= 1
            if idx == -1: break

        mask3 = np.zeros([args.maxlen], dtype=np.int32)  #
        idx3 = 0
        for idx in range(len(seq)):
            while idx3 < args.maxlen and t1[idx] >= t3[idx3]:
                idx3 += 1
            mask3[idx] = idx3

        for i in neg[u]:
            item_idx.append(i)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [seq2], [seq3], item_idx, [mask2], [mask3], "A"]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum, neg, user_train2, user_valid2, user_test2, itemnum2, neg2, user_train3, user_valid3, user_test3, itemnum3, neg3, time1, time2, time3, category, category_emb, User_Item, Item_User,
     category_contain] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    # if usernum>10000:
    #    users = random.sample(range(1, usernum + 1), 10000)
    # else:
    users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        t1 = np.zeros([args.maxlen], dtype=np.int32)  #
        t2 = np.zeros([args.maxlen], dtype=np.int32)  #
        t3 = np.zeros([args.maxlen], dtype=np.int32)  #
        idx = args.maxlen - 1
        for i, t in reversed(list(zip(train[u], time1[u]))):
            seq[idx] = i
            t1[idx] = t
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]

        seq2 = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i, t in reversed(list(zip(user_train2[u], time2[u]))):
            seq2[idx] = i
            t2[idx] = t
            idx -= 1
            if idx == -1: break

        mask2 = np.zeros([args.maxlen], dtype=np.int32)  #
        idx2 = 0
        for idx in range(len(seq)):
            while idx2 < args.maxlen and t1[idx] >= t2[idx2]:
                idx2 += 1
            mask2[idx] = idx2

        seq3 = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i, t in reversed(list(zip(user_train3[u], time3[u]))):
            seq3[idx] = i
            t3[idx] = t
            idx -= 1
            if idx == -1: break

        mask3 = np.zeros([args.maxlen], dtype=np.int32)  #
        idx3 = 0
        for idx in range(len(seq)):
            while idx3 < args.maxlen and t1[idx] >= t3[idx3]:
                idx3 += 1
            mask3[idx] = idx3

        for i in neg[u]:
            item_idx.append(i)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [seq2], [seq3], item_idx, [mask2], [mask3], "A"]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
#
# def evaluate2(model, dataset, args):
#     [train, valid, test, usernum, itemnum1, neg, train2, valid2, test2, itemnum2, neg2, time1, time2, category, category_emb, User_Item, Item_User,
#      category_contain] = copy.deepcopy(dataset)
#
#     NDCG = 0.0
#     HT = 0.0
#     valid_user = 0.0
#
#     # if usernum>10000:
#     #    users = random.sample(range(1, usernum + 1), 10000)
#     # else:
#     users = range(1, usernum + 1)
#     for u in users:
#
#         if len(train2[u]) < 1 or len(test2[u]) < 1: continue
#
#         seq2 = np.zeros([args.maxlen], dtype=np.int32)
#         t1 = np.zeros([args.maxlen], dtype=np.int32)  #
#         t2 = np.zeros([args.maxlen], dtype=np.int32)  #
#         idx = args.maxlen - 1
#         seq2[idx] = valid2[u][0]
#         idx -= 1
#         for i, t in reversed(list(zip(train2[u], time2[u]))):
#             seq2[idx] = i
#             t2[idx] = t
#             idx -= 1
#             if idx == -1: break
#         rated = set(train2[u])
#         rated.add(0)
#         item_idx = [test2[u][0]]
#
#         # for _ in range(100):
#         #     t = np.random.randint(1, itemnum + 1)
#         #     while t in rated: t = np.random.randint(1, itemnum + 1)
#         #     item_idx.append(t)
#
#         seq = np.zeros([args.maxlen], dtype=np.int32)
#         idx = args.maxlen - 1
#         for i, t in reversed(list(zip(train[u], time1[u]))):
#             seq[idx] = i
#             t1[idx] = t
#             idx -= 1
#             if idx == -1: break
#
#         mask = np.zeros([args.maxlen], dtype=np.int32)  #
#         idx2 = 0
#         for idx in range(len(seq2)):
#             while idx2 < args.maxlen and t2[idx] >= t1[idx2]:
#                 idx2 += 1
#             mask[idx] = idx2
#
#         for i in neg2[u]:
#             item_idx.append(i)
#
#         predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [seq2], item_idx, [mask], "B"]])
#         predictions = predictions[0]
#
#         rank = predictions.argsort().argsort()[0].item()
#
#         valid_user += 1
#
#         if rank < 10:
#             NDCG += 1 / np.log2(rank + 2)
#             HT += 1
#         if valid_user % 100 == 0:
#             print('.', end="")
#             sys.stdout.flush()
#
#     return NDCG / valid_user, HT / valid_user
#
#
# # evaluate on val set
# def evaluate_valid2(model, dataset, args):
#     [train, valid, test, usernum, itemnum, neg, train2, valid2, test2, itemnum2, neg2, time1, time2, category, category_emb, User_Item, Item_User,
#      category_contain] = copy.deepcopy(dataset)
#
#     NDCG = 0.0
#     valid_user = 0.0
#     HT = 0.0
#     # if usernum>10000:
#     #    users = random.sample(range(1, usernum + 1), 10000)
#     # else:
#     users = range(1, usernum + 1)
#     for u in users:
#         if len(train2[u]) < 1 or len(valid2[u]) < 1: continue
#
#         seq2 = np.zeros([args.maxlen], dtype=np.int32)
#         t1 = np.zeros([args.maxlen], dtype=np.int32)  #
#         t2 = np.zeros([args.maxlen], dtype=np.int32)  #
#         idx = args.maxlen - 1
#         for i, t in reversed(list(zip(train2[u], time2[u]))):
#             seq2[idx] = i
#             t2[idx] = t
#             idx -= 1
#             if idx == -1: break
#
#         rated = set(train2[u])
#         rated.add(0)
#         item_idx = [valid2[u][0]]
#
#         seq = np.zeros([args.maxlen], dtype=np.int32)
#         idx = args.maxlen - 1
#
#         for i, t in reversed(list(zip(train[u], time1[u]))):
#             seq[idx] = i
#             t1[idx] = t
#             idx -= 1
#             if idx == -1: break
#
#         mask = np.zeros([args.maxlen], dtype=np.int32)  #
#         idx2 = 0
#         for idx in range(len(seq2)):
#             while idx2 < args.maxlen and t2[idx] >= t1[idx2]:
#                 idx2 += 1
#             mask[idx] = idx2
#
#         for i in neg2[u]:
#             item_idx.append(i)
#
#         predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [seq2], item_idx, [mask], "B"]])
#         predictions = predictions[0]
#
#         rank = predictions.argsort().argsort()[0].item()
#
#         valid_user += 1
#
#         if rank < 10:
#             NDCG += 1 / np.log2(rank + 2)
#             HT += 1
#         if valid_user % 100 == 0:
#             print('.', end="")
#             sys.stdout.flush()
#
#     return NDCG / valid_user, HT / valid_user

def evaluate2(model, dataset, args):
    [train, valid, test, usernum, itemnum1, neg, user_train2, user_valid2, user_test2, itemnum2, neg2, user_train3, user_valid3, user_test3, itemnum3, neg3, time1, time2, time3, category, category_emb, User_Item, Item_User,
     category_contain] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    # if usernum>10000:
    #    users = random.sample(range(1, usernum + 1), 10000)
    # else:
    users = range(1, usernum + 1)
    for u in users:

        if len(user_train2[u]) < 1 or len(user_test2[u]) < 1: continue

        seq2 = np.zeros([args.maxlen], dtype=np.int32)
        t1 = np.zeros([args.maxlen], dtype=np.int32)  #
        t2 = np.zeros([args.maxlen], dtype=np.int32)  #
        t3 = np.zeros([args.maxlen], dtype=np.int32)  #
        idx = args.maxlen - 1
        seq2[idx] = user_valid2[u][0]
        idx -= 1
        for i, t in reversed(list(zip(user_train2[u], time2[u]))):
            seq2[idx] = i
            t2[idx] = t
            idx -= 1
            if idx == -1: break
        rated = set(user_train2[u])
        rated.add(0)
        item_idx = [user_test2[u][0]]

        # for _ in range(100):
        #     t = np.random.randint(1, itemnum + 1)
        #     while t in rated: t = np.random.randint(1, itemnum + 1)
        #     item_idx.append(t)

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i, t in reversed(list(zip(train[u], time1[u]))):
            seq[idx] = i
            t1[idx] = t
            idx -= 1
            if idx == -1: break

        mask1 = np.zeros([args.maxlen], dtype=np.int32)  #
        idx2 = 0
        for idx in range(len(seq2)):
            while idx2 < args.maxlen and t2[idx] >= t1[idx2]:
                idx2 += 1
            mask1[idx] = idx2

        seq3 = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i, t in reversed(list(zip(user_train3[u], time3[u]))):
            seq3[idx] = i
            t3[idx] = t
            idx -= 1
            if idx == -1: break

        mask3 = np.zeros([args.maxlen], dtype=np.int32)  #
        idx3 = 0
        for idx in range(len(seq)):
            while idx3 < args.maxlen and t2[idx] >= t3[idx3]:
                idx3 += 1
            mask3[idx] = idx3

        for i in neg2[u]:
            item_idx.append(i)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [seq2], [seq3], item_idx, [mask1], [mask3], "B"]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid2(model, dataset, args):
    [train, valid, test, usernum, itemnum, neg, user_train2, user_valid2, user_test2, itemnum2, neg2, user_train3, user_valid3, user_test3, itemnum3, neg3, time1, time2, time3, category, category_emb, User_Item, Item_User,
     category_contain] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    # if usernum>10000:
    #    users = random.sample(range(1, usernum + 1), 10000)
    # else:
    users = range(1, usernum + 1)
    for u in users:

        if len(user_train2[u]) < 1 or len(user_test2[u]) < 1: continue

        seq2 = np.zeros([args.maxlen], dtype=np.int32)
        t1 = np.zeros([args.maxlen], dtype=np.int32)  #
        t2 = np.zeros([args.maxlen], dtype=np.int32)  #
        t3 = np.zeros([args.maxlen], dtype=np.int32)  #
        idx = args.maxlen - 1
        # seq2[idx] = user_valid2[u][0]
        # idx -= 1
        for i, t in reversed(list(zip(user_train2[u], time2[u]))):
            seq2[idx] = i
            t2[idx] = t
            idx -= 1
            if idx == -1: break
        rated = set(user_train2[u])
        rated.add(0)
        item_idx = [user_valid2[u][0]]

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i, t in reversed(list(zip(train[u], time1[u]))):
            seq[idx] = i
            t1[idx] = t
            idx -= 1
            if idx == -1: break

        mask1 = np.zeros([args.maxlen], dtype=np.int32)  #
        idx2 = 0
        for idx in range(len(seq2)):
            while idx2 < args.maxlen and t2[idx] >= t1[idx2]:
                idx2 += 1
            mask1[idx] = idx2

        seq3 = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i, t in reversed(list(zip(user_train3[u], time3[u]))):
            seq3[idx] = i
            t3[idx] = t
            idx -= 1
            if idx == -1: break

        mask3 = np.zeros([args.maxlen], dtype=np.int32)  #
        idx3 = 0
        for idx in range(len(seq)):
            while idx3 < args.maxlen and t2[idx] >= t3[idx3]:
                idx3 += 1
            mask3[idx] = idx3

        for i in neg2[u]:
            item_idx.append(i)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [seq2], [seq3], item_idx, [mask1], [mask3], "B"]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate3(model, dataset, args):
    [train, valid, test, usernum, itemnum1, neg, user_train2, user_valid2, user_test2, itemnum2, neg2, user_train3, user_valid3, user_test3, itemnum3, neg3, time1, time2, time3, category, category_emb, User_Item, Item_User,
     category_contain] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    # if usernum>10000:
    #    users = random.sample(range(1, usernum + 1), 10000)
    # else:
    users = range(1, usernum + 1)
    for u in users:

        if len(user_train3[u]) < 1 or len(user_test3[u]) < 1: continue

        seq3 = np.zeros([args.maxlen], dtype=np.int32)
        t1 = np.zeros([args.maxlen], dtype=np.int32)  #
        t2 = np.zeros([args.maxlen], dtype=np.int32)  #
        t3 = np.zeros([args.maxlen], dtype=np.int32)  #
        idx = args.maxlen - 1
        seq3[idx] = user_valid3[u][0]
        idx -= 1
        for i, t in reversed(list(zip(user_train3[u], time3[u]))):
            seq3[idx] = i
            t3[idx] = t
            idx -= 1
            if idx == -1: break
        rated = set(user_train3[u])
        rated.add(0)
        item_idx = [user_test3[u][0]]

        # for _ in range(100):
        #     t = np.random.randint(1, itemnum + 1)
        #     while t in rated: t = np.random.randint(1, itemnum + 1)
        #     item_idx.append(t)

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i, t in reversed(list(zip(train[u], time1[u]))):
            seq[idx] = i
            t1[idx] = t
            idx -= 1
            if idx == -1: break

        mask1 = np.zeros([args.maxlen], dtype=np.int32)  #
        idx2 = 0
        for idx in range(len(seq3)):
            while idx2 < args.maxlen and t3[idx] >= t1[idx2]:
                idx2 += 1
            mask1[idx] = idx2

        seq2 = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i, t in reversed(list(zip(user_train2[u], time2[u]))):
            seq2[idx] = i
            t2[idx] = t
            idx -= 1
            if idx == -1: break

        mask2 = np.zeros([args.maxlen], dtype=np.int32)  #
        idx3 = 0
        for idx in range(len(seq3)):
            while idx3 < args.maxlen and t3[idx] >= t2[idx3]:
                idx3 += 1
            mask2[idx] = idx3

        for i in neg3[u]:
            item_idx.append(i)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [seq2], [seq3], item_idx, [mask1], [mask2], "C"]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid3(model, dataset, args):
    [train, valid, test, usernum, itemnum, neg, user_train2, user_valid2, user_test2, itemnum2, neg2, user_train3, user_valid3, user_test3, itemnum3, neg3, time1, time2, time3, category, category_emb, User_Item, Item_User,
     category_contain] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    # if usernum>10000:
    #    users = random.sample(range(1, usernum + 1), 10000)
    # else:
    users = range(1, usernum + 1)
    for u in users:

        if len(user_train3[u]) < 1 or len(user_test3[u]) < 1: continue

        seq3 = np.zeros([args.maxlen], dtype=np.int32)
        t1 = np.zeros([args.maxlen], dtype=np.int32)  #
        t2 = np.zeros([args.maxlen], dtype=np.int32)  #
        t3 = np.zeros([args.maxlen], dtype=np.int32)  #
        idx = args.maxlen - 1
        # seq3[idx] = user_valid3[u][0]
        # idx -= 1
        for i, t in reversed(list(zip(user_train3[u], time3[u]))):
            seq3[idx] = i
            t3[idx] = t
            idx -= 1
            if idx == -1: break
        rated = set(user_train3[u])
        rated.add(0)
        item_idx = [user_valid3[u][0]]

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i, t in reversed(list(zip(train[u], time1[u]))):
            seq[idx] = i
            t1[idx] = t
            idx -= 1
            if idx == -1: break

        mask1 = np.zeros([args.maxlen], dtype=np.int32)  #
        idx2 = 0
        for idx in range(len(seq3)):
            while idx2 < args.maxlen and t3[idx] >= t1[idx2]:
                idx2 += 1
            mask1[idx] = idx2

        seq2 = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i, t in reversed(list(zip(user_train2[u], time2[u]))):
            seq2[idx] = i
            t2[idx] = t
            idx -= 1
            if idx == -1: break

        mask2 = np.zeros([args.maxlen], dtype=np.int32)  #
        idx3 = 0
        for idx in range(len(seq3)):
            while idx3 < args.maxlen and t3[idx] >= t2[idx3]:
                idx3 += 1
            mask2[idx] = idx3

        for i in neg3[u]:
            item_idx.append(i)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [seq2], [seq3], item_idx, [mask1], [mask2], "C"]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

def get_category_adj(category, category2id, similarity_ex):
        category_num = len(category2id)
        # print(category_num)

        # A = np.zeros((category_num,category_num))  # 邻接矩阵
        N = np.zeros(category_num)  # ci出现次数
        M = np.zeros((category_num, category_num))  # ci和cj同时出现的次数

        for values in category.values():
            for idx, v in enumerate(values):  # 遍历list
                if v == 0:
                    continue
                N[v] += 1
                for j in range(idx + 1, len(values)):  # 比较i和list中i后面的每个元素
                    if values[j] != 0:
                        M[v][values[j]] += 1
                        M[values[j]][v] += 1    # 反过来也要加

        category_ex = {}
        for i in range(1, category_num):
            for j in range(1, category_num):
                p = M[i][j] / N[i]
                # score = 0.5 * similarity_ex[i][j] + p
                # if score >= 0.9:
                if p >= 0.5 or similarity_ex[i][j] >= 0.75:
                    if i in category_ex:
                        category_ex[i].append(j)
                    else:
                        category_ex[i] = [j]
                # A[i][j] = M[i][j] / N[i]
        # print(len(category_ex))
        # for k in category_ex.keys():
        #     print(category_ex[k])
        return category_ex
        # indices = []  # 存储每行中大于0.5的元素的下标以及它们所在的行的下标
        # for i, row in enumerate(A):
        #     row_indices = []  # 存储该行中大于0.5的元素的下标
        #     for j, val in enumerate(row):
        #         if val > 0.5:
        #             row_indices.append(j)
        #             indices.append([i, j])  # 将该元素的下标以及该行的下标存储到`indices`列表中
            # if len(row_indices) > 0:  # 如果该行中有大于0.5的元素，打印该行的下标以及这些元素的下标
                # s = next((key for key, value in category2id.items() if value == i), None)
                # print(s, end=": ")
                # for c in row_indices:
                #     cc = next((key for key, value in category2id.items() if value == c), None)
                #     print(cc, end=", ")
                # print()
                # print(f"Row {i}: {row_indices}")