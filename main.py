import os
import time
import numpy as np
import torch
import argparse
from tqdm import tqdm

from model import HGTL
from utils import *


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


parser = argparse.ArgumentParser()
parser.add_argument('--domain_A', required=True)
parser.add_argument('--domain_B', required=True)
parser.add_argument('--domain_C', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=100, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=500, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.001, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)

args = parser.parse_args()

if not os.path.isdir('result_' + args.train_dir):
    os.makedirs('result_' + args.train_dir)
with open(os.path.join('result_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))


def train_one_domain(sampler, model, bce_criterion, adam_optimizer, cl_weights, args, domain):
    u, seq, pos, neg, seq2, mask1, seq3, mask2 = sampler.next_batch()
    u, seq, pos, neg, seq2, mask1, seq3, mask2 = np.array(u), np.array(seq), np.array(pos), np.array(neg), np.array(
        seq2), np.array(mask1), np.array(seq3), np.array(mask2)
    pos_logits, neg_logits, con_loss1, con_loss2, con_loss3 = model(u, seq, seq2, seq3, pos, neg, mask1, mask2, domain)
    pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape,
                                                                                           device=args.device)
    adam_optimizer.zero_grad()
    indices = np.where(pos != 0)
    loss1 = bce_criterion(pos_logits[indices], pos_labels[indices])
    loss1 += bce_criterion(neg_logits[indices], neg_labels[indices])
    domain_loss = loss1 + cl_weights * con_loss1 + cl_weights * con_loss2 + cl_weights * con_loss3
    return domain_loss


def evaluate_and_log(model, dataset, args, f, domain, epoch, T):
    print('Evaluating', end='')
    if domain == "A":
        t_test = evaluate(model, dataset, args)
    elif domain == "B":
        t_test = evaluate2(model, dataset, args)
    elif domain == "C":
        t_test = evaluate3(model, dataset, args)
    t_valid = t_test
    print(
        f'{domain}: epoch:{epoch}, time: {T:.6f}(s), valid (NDCG@5: {t_valid[0]:.4f}, HR@5: {t_valid[1]:.4f}), test (NDCG@5: {t_test[0]:.4f}, HR@5: {t_test[1]:.4f}), valid (NDCG@10: {t_valid[2]:.4f}, HR@10: {t_valid[3]:.4f}), test (NDCG@10: {t_test[2]:.4f}, HR@10: {t_test[3]:.4f}), valid (NDCG@20: {t_valid[4]:.4f}, HR@20: {t_valid[5]:.4f}), test (NDCG@20: {t_test[4]:.4f}, HR@20: {t_test[5]:.4f})')
    f.write(str(t_valid) + ' ' + str(t_test) + '\n')
    f.flush()
    return time.time()


if __name__ == '__main__':
    dataset = data_partition(args.domain_A, args.domain_B, args.domain_C)
    [user_train1, user_valid1, user_test1, usernum, itemnum1, user_neg1, user_train2, user_valid2, user_test2, itemnum2,
     user_neg2, user_train3, user_valid3, user_test3, itemnum3, user_neg3, time1, time2, time3, category, category_emb,
     User_Item, Item_User, category_contain, category_num] = dataset
    num_batch = len(user_train1) // args.batch_size

    f = open(os.path.join('result_' + args.train_dir, 'log.txt'), 'w')

    samplers = [
        WarpSampler(user_train1, user_train2, user_train3, time1, time2, time3, usernum, [itemnum1, itemnum2, itemnum3],
                    batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3, target_index=0),
        WarpSampler(user_train1, user_train2, user_train3, time1, time2, time3, usernum, [itemnum1, itemnum2, itemnum3],
                    batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3, target_index=1),
        WarpSampler(user_train1, user_train2, user_train3, time1, time2, time3, usernum, [itemnum1, itemnum2, itemnum3],
                    batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3, target_index=2)
    ]
    model = HGTL(usernum, itemnum1 + itemnum2 + itemnum3, category, category_emb, category_num, User_Item, Item_User,
                 category_contain, args).to(args.device)

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
            import pdb
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
    with tqdm(total=args.num_epochs * num_batch, desc=f"Training", leave=False, ncols=100) as pbar:
        for epoch in range(epoch_start_idx, args.num_epochs + 1):
            if args.inference_only:
                break
            for step in range(num_batch):
                pbar.update(1)
                pbar.set_postfix({"Epoch": epoch})
                loss = 0
                for i, domain in enumerate(["A", "B", "C"]):
                    loss += train_one_domain(samplers[i], model, bce_criterion, adam_optimizer, cl_weights, args, domain)

                for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                loss.backward()
                adam_optimizer.step()

            if (epoch % 100 == 0) or (epoch % 10 == 0 and epoch >= 450):
                model.eval()
                for i, domain in enumerate(["A", "B", "C"]):
                    t1 = time.time() - t0
                    T += t1
                    t0 = evaluate_and_log(model, dataset, args, f, domain, epoch, T)
                model.train()

            if epoch == args.num_epochs:
                folder = 'result_' + args.train_dir
                fname = 'HGTL.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units,
                                     args.maxlen)
                torch.save(model.state_dict(), os.path.join(folder, fname))

    f.close()
    for sampler in samplers:
        sampler.close()
    print("Done")
