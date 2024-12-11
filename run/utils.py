import scipy.io as sio
import scipy.sparse as sp
import numpy as np
import networkx as nx
import random
import torch
import dgl
from torch_scatter import scatter
from torch_geometric.utils import to_dense_adj


def load_mat(dataset, train_rate=0.3, val_rate=0.1):
    """Load .mat dataset."""
    if dataset in ["books","reddit"]:
        data = torch.load("./dataset/{}.pt".format(dataset))
        network = to_dense_adj(data.edge_index)[0]
        network = np.array(network)
        attr = data.x
        attr = np.array(attr)
        label = data.y
        label = np.array(label)
        label[np.where(label > 0)] = 1

#将图的结构和节点的属性分别转换为适合稀疏存储的矩阵格式
        adj = sp.csr_matrix(network)
        feat = sp.lil_matrix(attr)
#检查是否有标签和异常标签
        if 'Class' in data:
            labels = np.squeeze(np.array(data['Class'], dtype=np.int64) - 1)
            num_classes = np.max(labels) + 1
            labels = dense_to_one_hot(labels, num_classes)
        else:
            labels = None

        ano_labels = np.squeeze(np.array(label))
        if 'str_anomaly_label' in data:
            str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
            attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
        else:
            str_ano_labels = None
            attr_ano_labels = None
#获取节点数量
        num_node = adj.shape[0]
#计算训练集和验证集的大小
        num_train = int(num_node * train_rate)
        num_val = int(num_node * val_rate)
#生成所有节点的索引
        all_idx = list(range(num_node))
#随机打乱节点索引，确保训练、验证和测试集的划分是随机的
        random.shuffle(all_idx)
        idx_train = all_idx[: num_train]
        idx_val = all_idx[num_train: num_train + num_val]
        idx_test = all_idx[num_train + num_val:]
    else:
#加载 .mat 文件
        data = sio.loadmat("./dataset/{}.mat".format(dataset))
#提取标签
        label = data['Label'] if ('Label' in data) else data['gnd']
#提取属性
        attr = data['Attributes'] if ('Attributes' in data) else data['X']
#提取网络（邻接矩阵）
        network = data['Network'] if ('Network' in data) else data['A']
#创建稀疏矩阵
        adj = sp.csr_matrix(network)
        feat = sp.lil_matrix(attr)
#处理类别标签
        labels = np.squeeze(np.array(data['Class'],dtype=np.int64) - 1)
        num_classes = np.max(labels) + 1
        labels = dense_to_one_hot(labels,num_classes)

        ano_labels = np.squeeze(np.array(label))
        if 'str_anomaly_label' in data:
            str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
            attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
        else:
            str_ano_labels = None
            attr_ano_labels = None

        num_node = adj.shape[0]
        num_train = int(num_node * train_rate)
        num_val = int(num_node * val_rate)
        all_idx = list(range(num_node))
        random.shuffle(all_idx)
        idx_train = all_idx[ : num_train]
        idx_val = all_idx[num_train : num_train + num_val]
        idx_test = all_idx[num_train + num_val : ]

    return adj, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels
#将类别标签转换为独热编码格式
def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
    return labels_one_hot
#生成一个图中节点的邻接信息
def generate_h_nodes_n_dict(adj, h):
    adj_h = sp.eye(adj.shape[0])
#到达其他节点的距离
    M = [{i: 0} for i in range(adj.shape[0])]
#每个节点在当前高度可以到达的节点
    h_index = [[i] for i in range(adj.shape[0])]
#迭代计算邻接信息
    for _ in range(h):
        if _ == 0:
            adj_h = sp.coo_matrix(adj_h * adj)

            # adj_h.row 稀疏矩阵行索引，adj_h.col 稀疏矩阵列索引
            for i, j in zip(adj_h.row, adj_h.col):
                if j in M[i]:
                    continue
                else:
                    M[i][j] = _ + 1
                    h_index[i].append(j)
        else:
            adj_h = sp.coo_matrix(adj_h * adj)

            # adj_h.row 稀疏矩阵行索引，adj_h.col 稀疏矩阵列索引
            for i, j in zip(adj_h.row, adj_h.col):
                if j in M[i]:
                    continue
                else:
                    M[i][j] = _ + 1
                    # h_index[i].append(j)
    # M 是字典，键是节点，值是源节点与该键值节点的距离
    # h_index 是h高度每个节点可达到的节点索引
    return M, h_index

#确保每个子图的节点数量达到预期，并包含源节点。它使用重试机制来处理节点数量不足的情况，适用于图数据分析和图神经网络的输入准备。
#通过随机游走重启（RWR）算法生成子图
def generate_rwr_subgraph(dgl_graph, subgraph_size):
    """Generate subgraph with RWR algorithm."""
    all_idx = list(range(dgl_graph.number_of_nodes()))
    reduced_size = subgraph_size - 1
#使用 DGL 提供的 random_walk_with_restart 方法，从所有节点开始进行随机游走重启，重启概率为 1，最多每个种子节点能访问 subgraph_size * 3 个节点。这个操作会生成多个游走路径。
    traces = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, all_idx, restart_prob=1, max_nodes_per_seed=subgraph_size*3)
#存储每个节点生成的子图节点
    subv = []
    for i,trace in enumerate(traces):
        subv.append(torch.unique(torch.cat(trace),sorted=False).tolist())
        retry_time = 0
        while len(subv[i]) < reduced_size:
            cur_trace = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, [i], restart_prob=0.9, max_nodes_per_seed=subgraph_size*5)
            subv[i] = torch.unique(torch.cat(cur_trace[0]),sorted=False).tolist()
            retry_time += 1
            if (len(subv[i]) <= 2) and (retry_time >10):
                subv[i] = (subv[i] * reduced_size)
        subv[i] = subv[i][:reduced_size]
        subv[i].append(i)

    return subv

#从给定的邻接信息中提取每个节点在不同跳数下的邻接节点，并将其存储在一个结构化的列表中
def get_n_hop_allsubgraph(M, hop_n):
    hop_n_allsubgraph = [[[]for _1 in range(hop_n)] for _ in range(len(M))]
    for i in range(len(M)):
        for key,value in M[i].items():
            hop_n_allsubgraph[i][value].append(key)
    return hop_n_allsubgraph

def get_n_hop_subgraph(hop_n_allsubgraph, sample_size):
    hop_n_subgraph = [[[] for _1 in range(len(hop_n_allsubgraph[0]))] for _ in range(len(hop_n_allsubgraph))]
    for i in range(len(hop_n_allsubgraph)):
        for j in range(1, len(hop_n_allsubgraph[i])):
            if len(hop_n_allsubgraph[i][j])>sample_size:
                 hop_n_subgraph[i][j] = random.sample(hop_n_allsubgraph[i][j], sample_size)
            else:
                hop_n_subgraph[i][j] = hop_n_allsubgraph[i][j]
                # hop_n_subgraph[i][j] = hop_n_allsubgraph[i][j]*sample_size
                # hop_n_subgraph[i][j] = hop_n_subgraph[i][j][:sample_size]
            # hop_n_subgraph[i][j].append(i)
    # hop_n_subgraph[源节点][n阶邻居]，不包括自身，自身在0阶邻居中
            hop_n_subgraph[i][j].append(i)
    return hop_n_subgraph

#将一个邻接矩阵（通常是稀疏形式）转换为 DGL 格式的图
def adj_to_dgl_graph(adj):
    """Convert adjacency matrix to dgl format."""
    nx_graph = nx.from_scipy_sparse_matrix(adj)
    dgl_graph = dgl.DGLGraph(nx_graph)
    return dgl_graph
#对图的特征矩阵进行行归一化，以确保每个节点的特征总和为 1。
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)
#将稀疏矩阵转换为元组表示形式
#选择是否插入批量维度？
def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx
#对邻接矩阵进行对称归一化处理
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
#将邻接矩阵转换为稀疏矩阵？
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
#从输入的 n 阶邻居信息中提取邻接矩阵和特征矩阵，并根据需要填充零行和零列，以确保输出矩阵的统一尺寸
def get_n_hop_np(hop_n_subgraph, adj, feature, sample_size_add1):
    hop_n_subadj=[[] for _ in range(len(hop_n_subgraph[0]))]
    hop_n_subft=[[] for c in range(len(hop_n_subgraph[0]))]
    for i in range(len(hop_n_subgraph)):
        for j in range(1, len(hop_n_subgraph[i])):
            cur_adj = adj[:, hop_n_subgraph[i][j], :][:, :, hop_n_subgraph[i][j]]
            cur_ft = feature[:, hop_n_subgraph[i][j], :]

            # cur_adj = torch.FloatTensor(cur_adj)
            # cur_ft = torch.FloatTensor(cur_ft)
            # cur_adj = cur_adj.cuda()
            # cur_ft = cur_ft.cuda()

            if len(hop_n_subgraph[i][j])<sample_size_add1:
                add_dim = sample_size_add1-len(hop_n_subgraph[i][j])
                orgin_dim = len(hop_n_subgraph[i][j])

                add_zero_row = torch.zeros((1, add_dim, orgin_dim))
                add_zero_row = add_zero_row.cuda()
                cur_adj = torch.cat((add_zero_row, cur_adj), dim=1)
                add_zero_col = torch.zeros((1, sample_size_add1, add_dim))
                add_zero_col = add_zero_col.cuda()
                cur_adj = torch.cat((cur_adj, add_zero_col), dim=2)

                for i in range(cur_adj.shape[1]):
                    cur_adj[:, i, i] = 1

                add_ft_row = torch.zeros((1, add_dim, feature.shape[2]))
                add_ft_row = add_ft_row.cuda()
                cur_ft = torch.cat((add_ft_row, cur_ft), dim=1)

            hop_n_subadj[j].append(cur_adj)
            hop_n_subft[j].append(cur_ft)
    #hop_n_subadj[n阶邻居][源节点]
    return hop_n_subadj, hop_n_subft

def cosine_similarity(tensor_A, tensor_B):
    # 计算点积
    dot_product = torch.sum(tensor_A * tensor_B, dim=1)

    # 计算范数
    norm_A = torch.norm(tensor_A, dim=1)
    norm_B = torch.norm(tensor_B, dim=1)

    # 将范数为零的位置设为一个小的非零值（避免除以零）
    norm_A[norm_A == 0] = 1e-7
    norm_B[norm_B == 0] = 1e-7

    # 计算余弦相似度
    similarity = dot_product / (norm_A * norm_B)

    return similarity

def generate_subgraph_embeding(attr, adj, subgraph_index, h):
    # 可能存在孤立节点，注意！
    subgraph_embeding = []
    for i in range(adj.shape[0]):
        root_feature = attr[i, :]                               #源点特征
        feature = attr[subgraph_index[i]]                       #子图属性矩阵
        feature = feature - np.tile(root_feature, (len(subgraph_index[i]), 1))
        adj_i = adj[subgraph_index[i], :][:, subgraph_index[i]] #子图邻接矩阵
        # WL
        subgraph_embeding.append(createWlEmbedding(feature, adj_i, h).reshape(1, -1))#WL

    #     邻居特征平均
    return subgraph_embeding
#实现 Weisfeiler-Lehman 方法生成图的节点嵌入，通过多次迭代聚合每个节点的邻居特征，并结合节点自身特征，最终得到一个表示节点的嵌入向量。
def createWlEmbedding(node_features, adj_mat, h):
    graph_feat = []


    for it in range(h+1):
        if it == 0:
            graph_feat.append(node_features)
        else:
            adj_cur = adj_mat+np.identity(adj_mat.shape[0])

            adj_cur = create_adj_avg(adj_cur)

            np.fill_diagonal(adj_cur, 0)

            #邻居信息聚合：np.dot(adj_cur, graph_feat[it-1]  自身信息：graph_feat[it-1]   类似与GraphSage
            graph_feat_cur = 0.5 * \
                (np.dot(adj_cur, graph_feat[it-1]) + graph_feat[it-1])
            # 将每次WL的嵌入都加入进去
            graph_feat.append(graph_feat_cur)
    #         先按列拼接，再按行求平均，得到的是

    return np.mean(np.concatenate(graph_feat, axis=1), axis=0)

def create_adj_avg(adj_cur):
    '''
    create adjacency
    '''
    deg = np.sum(adj_cur, axis=1)
    deg = np.asarray(deg).reshape(-1)

    deg[deg != 1] -= 1

    deg = 1/deg
    deg_mat = np.diag(deg)
    adj_cur = adj_cur.dot(deg_mat.T).T

    return adj_cur

def get_sim_matrix(subgraph_embeding):
    # 将所有矩阵按行排列成一个大矩阵
    mat_array = np.concatenate(subgraph_embeding, axis=0)
    # print(mat_array.shape)
    # 计算所有向量的模长
    norms = np.linalg.norm(mat_array, axis=1)
    # print(len(norms))
    c = np.outer(norms, norms)
    # print(c.shape)
    # 计算余弦相似度矩阵
    norms[norms==0] = np.inf
    sim_matrix = np.dot(mat_array, mat_array.T) / np.outer(norms, norms)
    # sim_matrix[np.isnan(sim_matrix)] = 0
    sim_matrix = torch.FloatTensor(sim_matrix)
    return sim_matrix
    # print(sim_matrix.shape)
    # print(sim_matrix)

def get_new_adj(similarity_matrix, start_adj, add_value=0.8, delete_value=0.3):
    new_adj = start_adj.copy()
    # 使用PyTorch张量操作函数替代for循环，加速运算
    indices = torch.nonzero(torch.gt(similarity_matrix, add_value) & torch.ne(torch.eye(similarity_matrix.size(0)), 1),
                            as_tuple=False)
    new_adj[indices[:, 0], indices[:, 1]] = 1
    indices = torch.nonzero(torch.lt(similarity_matrix, delete_value) & torch.ne(torch.eye(similarity_matrix.size(0)), 1),
                            as_tuple=False)
    new_adj[indices[:, 0], indices[:, 1]] = 0

    return new_adj