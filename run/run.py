import random

import torch
import torch.nn as nn
from model import Model
from utils import *
import argparse
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser(description='method1: Self-Supervised Contrastive Learning for Anomaly Detection')
parser.add_argument('--dataset', type=str, default='books')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--alpha', type=float, default=0.7)
parser.add_argument('--num_epoch', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--sample_size', type=int, default=3)
parser.add_argument('--hop_n', type=int, default=3)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--readout', type=str, default='avg')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--auc_test_rounds', type=int, default=20)
parser.add_argument('--balance', type=list, default=[0.5,0.3,0.2])
parser.add_argument('--strlist', type=str, default='[1,1,1]')
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
subgraph_size = args.sample_size + 1


# Set random seed
dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


print('Dataset: ', args.dataset)
# Load and preprocess data

adj, features, labels, idx_train, idx_val,\
idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)
adj_orgin = adj.copy()
features_orgin = features.copy()


dgl_graph = adj_to_dgl_graph(adj)

nb_nodes = features.shape[0]
ft_size = features.shape[1]

features, _ = preprocess_features(features)
adj = normalize_adj(adj)
adj = (adj + sp.eye(adj.shape[0])).todense()

features = torch.FloatTensor(features[np.newaxis])
adj = torch.FloatTensor(adj[np.newaxis])
if labels!=None:
    labels = torch.FloatTensor(labels[np.newaxis])
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

M, h_index = generate_h_nodes_n_dict(adj_orgin, h=args.hop_n)
hop_n_allsubgraph = get_n_hop_allsubgraph(M, hop_n=args.hop_n+1)


# add_adj部分

subgraph_embeding = generate_subgraph_embeding(attr=features_orgin.A, adj=adj_orgin.A, subgraph_index=h_index, h=1)
# 通过子图向量得到每个节点的相似矩阵
sim_matrix = get_sim_matrix(subgraph_embeding)

# 得到初始修改后的邻接矩阵
add_adj = get_new_adj(sim_matrix, start_adj=adj_orgin.A, add_value=0.7, delete_value=-0.1)
# 对add_adj进行节点选取
add_M, add_h_index = generate_h_nodes_n_dict(add_adj, h=args.hop_n)
add_hop_n_allsubgraph = get_n_hop_allsubgraph(add_M, hop_n=args.hop_n+1)
# add_adj正则化
add_adj = normalize_adj(add_adj+sp.eye(add_adj.shape[0]))
add_adj= add_adj.toarray()
# 转换张量
add_adj= torch.FloatTensor(add_adj[np.newaxis])



# Initialize model and optimiser
model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    if labels != None:
        labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

    add_adj = add_adj.cuda()


if torch.cuda.is_available():
    # 二分类交叉熵损失
    b_xent_ns = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).cuda())
    b_xent_nn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).cuda())
    add_b_xent_ns = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).cuda())
    add_b_xent_nn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).cuda())
else:
    b_xent_ns = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))
    b_xent_nn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))
    add_b_xent_ns = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))
    add_b_xent_nn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))


#交叉熵损失
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
# 最佳训练轮次
best_t = 0
# 多少批次
batch_num = nb_nodes // args.batch_size + 1


# Train model:提供可视化进度反馈
with tqdm(total=args.num_epoch) as pbar:
    pbar.set_description('Training')
    for epoch in range(args.num_epoch):
        loss_full_batch = torch.zeros((nb_nodes, 1))
        if torch.cuda.is_available():
            loss_full_batch = loss_full_batch.cuda()

        model.train()

        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)
        total_loss = 0

        hop_n_subgraph = get_n_hop_subgraph(hop_n_allsubgraph, args.sample_size)
        hop_n_subadj, hop_n_subft = get_n_hop_np(hop_n_subgraph, adj, features, sample_size_add1=args.sample_size+1)

        add_hop_n_subgraph = get_n_hop_subgraph(add_hop_n_allsubgraph, args.sample_size)
        add_hop_n_subadj, add_hop_n_subft = get_n_hop_np(add_hop_n_subgraph, add_adj, features, sample_size_add1=args.sample_size + 1)

        for batch_idx in range(batch_num):

            optimiser.zero_grad()

            is_final_batch = (batch_idx == (batch_num - 1))

            if not is_final_batch:
                idx = all_idx[batch_idx * args.batch_size: (batch_idx + 1) * args.batch_size]
            else:
                idx = all_idx[batch_idx * args.batch_size:]

            cur_batch_size = len(idx)

            lbl_ns = torch.unsqueeze(
                torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio))), 1).to(device)

            lbl_nn= torch.unsqueeze(torch.cat(
                (torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio))), 1).to(device)

            ba = [[] for i in range(args.hop_n+1)]
            bf = [[] for i in range(args.hop_n+1)]

            add_ba = [[] for i in range(args.hop_n + 1)]
            add_bf = [[] for i in range(args.hop_n + 1)]

            added_adj_zero_row = torch.zeros((cur_batch_size, 1, args.sample_size+1))
            added_adj_zero_col = torch.zeros((cur_batch_size, args.sample_size + 2, 1))
            added_adj_zero_col[:, -1, :] = 1.
            added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))


            if torch.cuda.is_available():
                added_adj_zero_row = added_adj_zero_row.cuda()
                added_adj_zero_col = added_adj_zero_col.cuda()
                added_feat_zero_row = added_feat_zero_row.cuda()


            for i in idx:
                for j in range(1, args.hop_n+1):
                    ba[j].append(hop_n_subadj[j][i])
                    bf[j].append(hop_n_subft[j][i])

                    add_ba[j].append(add_hop_n_subadj[j][i])
                    add_bf[j].append(add_hop_n_subft[j][i])

            for i in range(1, args.hop_n+1):
                ba[i] = torch.cat(ba[i])
                ba[i] = torch.cat((ba[i], added_adj_zero_row), dim=1)
                ba[i] = torch.cat((ba[i], added_adj_zero_col), dim=2)

                add_ba[i] = torch.cat(add_ba[i])
                add_ba[i] = torch.cat((add_ba[i], added_adj_zero_row), dim=1)
                add_ba[i] = torch.cat((add_ba[i], added_adj_zero_col), dim=2)

                bf[i] = torch.cat(bf[i])
                bf[i] = torch.cat((bf[i][:, :-1, :], added_feat_zero_row, bf[i][:, -1:, :]), dim=1)

                add_bf[i] = torch.cat(add_bf[i])
                add_bf[i] = torch.cat((add_bf[i][:, :-1, :], added_feat_zero_row, add_bf[i][:, -1:, :]), dim=1)

            logits_ns, logits_nn, subgraph_embed_list, node_embed_list= model(bf, ba)
            add_logits_ns, add_logits_nn, add_subgraph_embed_list, add_node_embed_list = model(add_bf, add_ba)

            # subgraph-subgraph contrast loss
            NCE_loss_list = []
            for i in range(len(subgraph_embed_list)):
                subgraph_embed = F.normalize(subgraph_embed_list[i], dim=1, p=2)
                subgraph_embed_hat = F.normalize(add_subgraph_embed_list[i], dim=1, p=2)
                sim_matrix_one = torch.matmul(subgraph_embed, subgraph_embed_hat.t())
                sim_matrix_two = torch.matmul(subgraph_embed, subgraph_embed.t())
                sim_matrix_three = torch.matmul(subgraph_embed_hat, subgraph_embed_hat.t())
                temperature = 1.0
                sim_matrix_one_exp = torch.exp(sim_matrix_one / temperature)
                sim_matrix_two_exp = torch.exp(sim_matrix_two / temperature)
                sim_matrix_three_exp = torch.exp(sim_matrix_three / temperature)
                nega_list = np.arange(0, cur_batch_size - 1, 1)
                nega_list = np.insert(nega_list, 0, cur_batch_size - 1)
                sim_row_sum = sim_matrix_one_exp[:, nega_list] + sim_matrix_two_exp[:, nega_list] + sim_matrix_three_exp[:,
                                                                                                    nega_list]
                sim_row_sum = torch.diagonal(sim_row_sum)
                sim_diag = torch.diagonal(sim_matrix_one)
                sim_diag_exp = torch.exp(sim_diag / temperature)
                NCE_loss = -torch.log(sim_diag_exp / (sim_row_sum))
                NCE_loss = torch.mean(NCE_loss)
                NCE_loss_list.append(NCE_loss)

            loss_ns_list = [[] for i in range(len(logits_ns))]
            loss_nn_list = [[] for i in range(len(logits_nn))]

            add_loss_ns_list = [[] for i in range(len(add_logits_ns))]
            add_loss_nn_list = [[] for i in range(len(add_logits_nn))]
            for i in range(len(logits_ns)):
                loss_ns_list[i] = b_xent_ns(logits_ns[i], lbl_ns)
                loss_nn_list[i] = b_xent_ns(logits_nn[i], lbl_nn)

                add_loss_ns_list[i] = add_b_xent_ns(add_logits_ns[i], lbl_ns)
                add_loss_nn_list[i] = add_b_xent_ns(add_logits_nn[i], lbl_nn)

            loss_ns=0
            loss_nn=0

            add_loss_ns = 0
            add_loss_nn = 0
            NCE_loss=0
            for i in range(len(loss_ns_list)):
                loss_ns = loss_ns_list[i]*args.balance[i]+loss_ns
                loss_nn = loss_nn_list[i] * args.balance[i] + loss_nn

                add_loss_ns = add_loss_ns_list[i] * args.balance[i] + add_loss_ns
                add_loss_nn = add_loss_nn_list[i] * args.balance[i] + add_loss_nn

                NCE_loss = NCE_loss_list[i]*args.balance[i]+NCE_loss

            loss = args.alpha*loss_ns+(1-args.alpha)*loss_nn + args.alpha*add_loss_ns+(1-args.alpha)*add_loss_nn
            loss = torch.mean(loss)
            loss.backward()
            optimiser.step()

            loss = loss.detach().cpu().numpy()
            if not is_final_batch:
                total_loss += loss

            mean_loss = (total_loss * args.batch_size + loss * cur_batch_size) / nb_nodes

            if mean_loss < best:
                best = mean_loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), 'best_model.pkl')
            else:
                cnt_wait += 1

            # print('Epoch:{} Loss:{:.8f}'.format(epoch, mean_loss), flush=True)

        pbar.set_postfix(loss=mean_loss)
        pbar.update(1)

end = time.time()
# print('%.2f%', end-start)

# Test model
print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load('best_model.pkl'))

multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))
multi_round_ano_score_p = np.zeros((args.auc_test_rounds, nb_nodes))
multi_round_ano_score_n = np.zeros((args.auc_test_rounds, nb_nodes))


with tqdm(total=args.auc_test_rounds) as pbar_test:
    pbar_test.set_description('Testing')
    for round in range(args.auc_test_rounds):
        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)
        total_loss = 0

        hop_n_subgraph = get_n_hop_subgraph(hop_n_allsubgraph, args.sample_size)
        hop_n_subadj, hop_n_subft = get_n_hop_np(hop_n_subgraph, adj, features, sample_size_add1=args.sample_size + 1)

        add_hop_n_subgraph = get_n_hop_subgraph(add_hop_n_allsubgraph, args.sample_size)
        add_hop_n_subadj, add_hop_n_subft = get_n_hop_np(add_hop_n_subgraph, add_adj, features, sample_size_add1=args.sample_size + 1)

        for batch_idx in range(batch_num):

            optimiser.zero_grad()

            is_final_batch = (batch_idx == (batch_num - 1))

            if not is_final_batch:
                idx = all_idx[batch_idx * args.batch_size: (batch_idx + 1) * args.batch_size]
            else:
                idx = all_idx[batch_idx * args.batch_size:]

            cur_batch_size = len(idx)

            lbl_ns = torch.unsqueeze(
                torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio))), 1).to(device)

            lbl_nn = torch.unsqueeze(torch.cat(
                (torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio))), 1).to(device)

            ba = [[] for i in range(args.hop_n + 1)]
            bf = [[] for i in range(args.hop_n + 1)]

            add_ba = [[] for i in range(args.hop_n + 1)]
            add_bf = [[] for i in range(args.hop_n + 1)]

            added_adj_zero_row = torch.zeros((cur_batch_size, 1, args.sample_size + 1))
            added_adj_zero_col = torch.zeros((cur_batch_size, args.sample_size + 2, 1))
            added_adj_zero_col[:, -1, :] = 1.
            added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

            if torch.cuda.is_available():
                added_adj_zero_row = added_adj_zero_row.cuda()
                added_adj_zero_col = added_adj_zero_col.cuda()
                added_feat_zero_row = added_feat_zero_row.cuda()

            for i in idx:
                for j in range(1, args.hop_n + 1):
                    ba[j].append(hop_n_subadj[j][i])
                    bf[j].append(hop_n_subft[j][i])

                    add_ba[j].append(add_hop_n_subadj[j][i])
                    add_bf[j].append(add_hop_n_subft[j][i])

            for i in range(1, args.hop_n + 1):
                ba[i] = torch.cat(ba[i])
                ba[i] = torch.cat((ba[i], added_adj_zero_row), dim=1)
                ba[i] = torch.cat((ba[i], added_adj_zero_col), dim=2)

                add_ba[i] = torch.cat(add_ba[i])
                add_ba[i] = torch.cat((add_ba[i], added_adj_zero_row), dim=1)
                add_ba[i] = torch.cat((add_ba[i], added_adj_zero_col), dim=2)

                bf[i] = torch.cat(bf[i])
                bf[i] = torch.cat((bf[i][:, :-1, :], added_feat_zero_row, bf[i][:, -1:, :]), dim=1)

                add_bf[i] = torch.cat(add_bf[i])
                add_bf[i] = torch.cat((add_bf[i][:, :-1, :], added_feat_zero_row, add_bf[i][:, -1:, :]), dim=1)

            with torch.no_grad():
                logits_ns, logits_nn, subgraph_embed_list, node_embed_list = model(bf, ba)
                add_logits_ns, add_logits_nn, add_subgraph_embed_list, add_node_embed_list = model(add_bf, add_ba)

                for i in range(len(logits_ns)):
                    logits_ns[i] = torch.squeeze(logits_ns[i])
                    logits_nn[i] = torch.squeeze(logits_nn[i])
                    logits_ns[i] = torch.sigmoid(logits_ns[i])
                    logits_nn[i] = torch.sigmoid(logits_nn[i])

                    add_logits_ns[i] = torch.squeeze(add_logits_ns[i])
                    add_logits_nn[i] = torch.squeeze(add_logits_nn[i])
                    add_logits_ns[i] = torch.sigmoid(add_logits_ns[i])
                    add_logits_nn[i] = torch.sigmoid(add_logits_nn[i])

            ano_score_list_ns = [[] for i in range(len(logits_ns))]
            ano_score_list_nn = [[] for i in range(len(logits_nn))]

            add_ano_score_list_ns = [[] for i in range(len(add_logits_ns))]
            add_ano_score_list_nn = [[] for i in range(len(add_logits_nn))]

            for i in range(len(logits_ns)):
                ano_score_list_ns[i] = - (logits_ns[i][:cur_batch_size] - logits_ns[i][cur_batch_size:]).cpu().numpy()
                ano_score_list_nn[i] = - (logits_nn[i][:cur_batch_size] - logits_nn[i][cur_batch_size:]).cpu().numpy()

                add_ano_score_list_ns[i] = - (add_logits_ns[i][:cur_batch_size] - add_logits_ns[i][cur_batch_size:]).cpu().numpy()
                add_ano_score_list_nn[i] = - (add_logits_nn[i][:cur_batch_size] - add_logits_nn[i][cur_batch_size:]).cpu().numpy()

            ano_score_ns = 0
            ano_score_nn = 0
            add_ano_score_ns = 0
            add_ano_score_nn = 0
            for i in range(len(ano_score_list_ns)):
                ano_score_ns = ano_score_list_ns[i]*args.balance[i]+ano_score_ns
                ano_score_nn = ano_score_list_nn[i] * args.balance[i] + ano_score_nn

                add_ano_score_ns = add_ano_score_list_ns[i] * args.balance[i] + add_ano_score_ns
                add_ano_score_nn = add_ano_score_list_nn[i] * args.balance[i] + add_ano_score_nn

            ano_score = args.alpha*ano_score_ns + (1-args.alpha)*ano_score_nn + args.alpha*add_ano_score_ns + (1-args.alpha)*add_ano_score_nn

            multi_round_ano_score[round, idx] = ano_score

        pbar_test.update(1)

ano_score_final = np.mean(multi_round_ano_score, axis=0)
# ano_score_final_p = np.mean(multi_round_ano_score_p, axis=0)
# ano_score_final_n = np.mean(multi_round_ano_score_n, axis=0)
auc = roc_auc_score(ano_label, ano_score_final)

data = pd.DataFrame({
    'Label': ano_label,
    'Anomaly Score': ano_score_final
})
data.to_csv('label_score_data.csv', index=False)


plt.plot(*roc_curve(ano_label, ano_score_final)[:2], label='GRADE')
plt.show()
print('AUC:{:.4f}'.format(auc))