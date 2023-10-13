import os
import time

import numpy as np
import torch
import argparse

from model_triple import SASRec
from utils_triple import *


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=100, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=501, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.001, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    dataset = data_partition(args.dataset, "CDs_and_Vinyl", "Books")

    [user_train1, user_valid1, user_test1, usernum, itemnum1, user_neg1, user_train2, user_valid2, user_test2, itemnum2, user_neg2,
     user_train3, user_valid3, user_test3, itemnum3, user_neg3, time1, time2, time3, category, category_emb, User_Item, Item_User, category_contain] = dataset
    num_batch = len(user_train1) // args.batch_size
    cc = 0.0
    for u in user_train1:
        cc += len(user_train1[u])
    print('average sequence length: %.2f' % (cc / len(user_train1)))

    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log1.9.txt'), 'w')

    sampler1 = WarpSampler1(user_train1, user_train2, user_train3, time1, time2, time3, usernum, itemnum1, itemnum2, itemnum3, batch_size=args.batch_size,
                          maxlen=args.maxlen, n_workers=3)
    sampler2 = WarpSampler2(user_train1, user_train2, user_train3, time1, time2, time3, usernum, itemnum1, itemnum2, itemnum3, batch_size=args.batch_size,
                            maxlen=args.maxlen, n_workers=3)
    sampler3 = WarpSampler3(user_train1, user_train2, user_train3, time1, time2, time3, usernum, itemnum1, itemnum2,
                            itemnum3, batch_size=args.batch_size,
                            maxlen=args.maxlen, n_workers=3)
    model = SASRec(usernum, itemnum1 + itemnum2 + itemnum3, category, category_emb, User_Item, Item_User, category_contain,
                   args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass
    model.train()

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb;

            pdb.set_trace()

    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))

    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    cl_weights = 0.5

    T = 0.0
    t0 = time.time()
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break
        for step in range(num_batch):  # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            A_u, A_seq, A_pos, A_neg, A_seq2, A_mask2, A_seq3, A_mask3 = sampler1.next_batch()  # tuples to ndarray
            A_u, A_seq, A_pos, A_neg, A_seq2, A_mask2, A_seq3, A_mask3 = np.array(A_u), np.array(A_seq), np.array(A_pos), np.array(A_neg), np.array(
                A_seq2), np.array(A_mask2), np.array(A_seq3), np.array(A_mask3)

            A_pos_logits, A_neg_logits, A_con_loss1, A_con_loss2, A_con_loss3 = model(A_u, A_seq, A_seq2, A_seq3, A_pos, A_neg, A_mask2, A_mask3, "A")
            A_pos_labels, A_neg_labels = torch.ones(A_pos_logits.shape, device=args.device), torch.zeros(A_neg_logits.shape,
                                                                                                   device=args.device)

            adam_optimizer.zero_grad()
            indices = np.where(A_pos != 0)
            A_loss1 = bce_criterion(A_pos_logits[indices], A_pos_labels[indices])
            A_loss1 += bce_criterion(A_neg_logits[indices], A_neg_labels[indices])
            loss = A_loss1 + cl_weights * A_con_loss1 + cl_weights * A_con_loss2 + cl_weights * A_con_loss3

            # domain2
            B_u, B_seq, B_pos, B_neg, B_seq2, B_mask1, B_seq3, B_mask3 = sampler2.next_batch()
            B_u, B_seq, B_pos, B_neg, B_seq2, B_mask1, B_seq3, B_mask3 = np.array(B_u), np.array(B_seq), np.array(B_pos), np.array(B_neg), np.array(
                B_seq2), np.array(B_mask1), np.array(B_seq3), np.array(B_mask3)
            B_pos_logits, B_neg_logits, B_con_loss1, B_con_loss2, B_con_loss3 = model(B_u, B_seq, B_seq2, B_seq3, B_pos, B_neg, B_mask1, B_mask3, "B")
            B_pos_labels, B_neg_labels = torch.ones(B_pos_logits.shape, device=args.device), torch.zeros(B_neg_logits.shape,
                                                                                                   device=args.device)

            adam_optimizer.zero_grad()
            indices = np.where(B_pos != 0)
            B_loss1 = bce_criterion(B_pos_logits[indices], B_pos_labels[indices])
            B_loss1 += bce_criterion(B_neg_logits[indices], B_neg_labels[indices])
            loss += B_loss1 + cl_weights * B_con_loss1 + cl_weights * B_con_loss2 + cl_weights * B_con_loss3

            # domain3
            C_u, C_seq, C_pos, C_neg, C_seq2, C_mask1, C_seq3, C_mask2 = sampler3.next_batch()
            C_u, C_seq, C_pos, C_neg, C_seq2, C_mask1, C_seq3, C_mask2 = np.array(C_u), np.array(C_seq), np.array(C_pos), np.array(C_neg), np.array(
                C_seq2), np.array(C_mask1), np.array(C_seq3), np.array(C_mask2)
            C_pos_logits, C_neg_logits, C_con_loss1, C_con_loss2, C_con_loss3 = model(C_u, C_seq, C_seq2, C_seq3, C_pos, C_neg, C_mask1, C_mask2, "C")
            C_pos_labels, C_neg_labels = torch.ones(C_pos_logits.shape, device=args.device), torch.zeros(C_neg_logits.shape,
                                                                                                   device=args.device)
            adam_optimizer.zero_grad()
            indices = np.where(C_pos != 0)
            C_loss1 = bce_criterion(C_pos_logits[indices], C_pos_labels[indices])
            C_loss1 += bce_criterion(C_neg_logits[indices], C_neg_labels[indices])
            loss += C_loss1 + cl_weights * C_con_loss1 + cl_weights * C_con_loss2 + cl_weights * C_con_loss3

            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()

        if (epoch % 20 == 0 and epoch > 400) or epoch == 10:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test1 = evaluate(model, dataset, args)
            t_valid1 = evaluate_valid(model, dataset, args)
            print('domain A: epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                  % (epoch, T, t_valid1[0], t_valid1[1], t_test1[0], t_test1[1]))

            f.write(str(t_valid1) + ' ' + str(t_test1) + '\n')
            f.flush()
            t0 = time.time()

            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test2 = evaluate2(model, dataset, args)
            t_valid2 = evaluate_valid2(model, dataset, args)
            print('domain B: epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                  % (epoch, T, t_valid2[0], t_valid2[1], t_test2[0], t_test2[1]))

            f.write(str(t_valid2) + ' ' + str(t_test2) + '\n')
            f.flush()
            t0 = time.time()

            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test3 = evaluate3(model, dataset, args)
            t_valid3 = evaluate_valid3(model, dataset, args)
            print('domain C: epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                  % (epoch, T, t_valid3[0], t_valid3[1], t_test3[0], t_test3[1]))

            f.write(str(t_valid3) + ' ' + str(t_test3) + '\n')
            f.flush()
            t0 = time.time()

            model.train()

        if epoch == args.num_epochs:
            folder = args.dataset + '_' + args.train_dir
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units,
                                 args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))

    f.close()
    sampler1.close()
    sampler2.close()
    sampler3.close()
    print("Done")
