import pickle
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
import h5py
from tqdm import tqdm

MAP_CUTOFF = 6
INPUT_DIM = 64
HIDDEN_DIM = 512
NLAYER = 3
DROPOUT = 0.1
LEARNING_RATE = 5E-5
BATCH_SIZE = 1
NUM_CLASSES = 2
NUMBER_EPOCHS = 500
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_EC1_num(ec_onehot):
    ec1 = list(int(i) for i in ec_onehot)
    ec_index = ec1.index(1.0)
    if ec_index == 0:
        EC_index = 7
    elif ec_index == 1:
        EC_index = 3
    elif ec_index == 2:
        EC_index = 4
    elif ec_index == 3:
        EC_index = 5
    elif ec_index == 4:
        EC_index = 6
    elif ec_index == 5:
        EC_index = 1
    else:
        EC_index = 2
    return EC_index

def remove_nan(matrix,padding_value=0.):
    aa_has_nan = np.isnan(matrix).reshape([len(matrix),-1]).max(-1)
    matrix[aa_has_nan] = padding_value
    return matrix

def get_cluster_center(ec2id, data_path, epoch):
    cluster_center = {}
    if epoch != 0:
        f_read = open(data_path+'updated_enzfeas/dict_enzfeas.pkl',"rb")
        dict_enzfeas = pickle.load(f_read)
        for ec in tqdm(list(ec2id.keys())):
            avg_pro_feas = []
            for pro_id in ec2id[ec]:
                if pro_id in dict_enzfeas.keys():
                    pro_feature = dict_enzfeas[pro_id][:]
                    avg_pro_feas.append(torch.squeeze(pro_feature))
            cluster_center[ec] = torch.stack(avg_pro_feas,dim=0).mean(dim=0)
    else:
        with h5py.File(data_path + 'Prot5/train_per_protein_embeddings.h5', "r") as f:
            for ec in tqdm(list(ec2id.keys())):
                avg_pro_feas = []
                for pro_id in ec2id[ec]:
                    pro_feature = f[pro_id][:]
                    avg_pro_feas.append(pro_feature)
                cluster_center[ec] = torch.from_numpy(np.mean(avg_pro_feas, axis=0))
                cluster_center[ec] = torch.unsqueeze(cluster_center[ec],0)
    return cluster_center


def embedding(sequence_name, data_path):
    pssm_feature = np.load(data_path + "pssm/processed_pssm/" + sequence_name + '.npy')
    hmm_feature = np.load(data_path + "hhm/processed_hhm/" + sequence_name + '.npy')
    with h5py.File(data_path + 'Prot5/train_per_residue_embeddings.h5', "r") as f:
        evo_feature = f[sequence_name][:]
    return pssm_feature, hmm_feature, evo_feature.astype(np.float32)


def get_atom_features(sequence_name, data_path):
    atom_feature = np.load(data_path + 'Atom_feas/matched_atomfea/'+ sequence_name + '.npy')
    atom_feature = remove_nan(atom_feature, padding_value=0.)

    seq_feature = np.load(data_path + 'seqfea/' + sequence_name + '.npy')
    return atom_feature, seq_feature

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** 0.5).flatten()
    for i in range(len(r_inv)):
        r_inv[i] = 0 if r_inv[i] == 0 else 1 / r_inv[i]
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result

def load_graph(sequence_name, data_path):
    #using PDB structure
    dismap = np.load(data_path + "contact_map/" + sequence_name + ".npy")
    mask = ((dismap >= 0) * (dismap <= MAP_CUTOFF))
    adjacency_matrix = mask.astype(np.int)
    norm_matrix = normalize(adjacency_matrix.astype(np.float32))
    return norm_matrix


class ProDataset(Dataset):
    def __init__(self, dataframe, data_path):
        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values
        self.EC1 = dataframe['EC1'].values
        self.data_path = data_path

    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        label = np.array(self.labels[index])
        EC1 = np.array(self.EC1[index])
        data_path = self.data_path

        pssm_feature, hmm_feature, evo_feature = embedding(sequence_name, data_path)
        atom_features, seq_feature = get_atom_features(sequence_name, data_path)
        node_features = np.concatenate([pssm_feature, hmm_feature, atom_features, seq_feature], axis=1)
        graph = load_graph(sequence_name, data_path)

        return sequence_name, sequence, label, EC1.astype(np.int), node_features, graph, evo_feature

    def __len__(self):
        return len(self.labels)

class GINLayer(nn.Module):
    def __init__(self, nhidden):
        super(GINLayer, self).__init__()
        self.linear1 = nn.Linear(nhidden, nhidden)
        self.linear2 = nn.Linear(nhidden, nhidden)
        self.relu = nn.ReLU()

    def forward(self, node_feat, adj ):
        neighbor_agg = torch.matmul(adj, node_feat)
        h = self.relu(self.linear1((1 + 0.1) * node_feat + neighbor_agg))
        h = self.linear2(h)
        return h


class GraphAttentionLayer(nn.Module):
    def __init__(self, nhidden):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = nhidden
        self.out_features = nhidden

        # Learnable parameters: attention mechanism
        self.W = nn.Parameter(torch.zeros(size=(nhidden, nhidden)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * nhidden, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)


    def forward(self, input, adj):
        h = torch.matmul(input, self.W)  # Linear transformation

        # Self-attention mechanism
        N = h.size()[0]  # Number of nodes
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # Masked attention scores
        attention = F.softmax(attention, dim=1)  # Attention coefficients
        h_prime = torch.matmul(attention, h)  # Linear combination using attention scores

        return h_prime

class GraphConvolution(nn.Module):
    def __init__(self, nhidden):
        super(GraphConvolution, self).__init__()

        self.nhidden = nhidden
        self.projection = nn.Linear(self.nhidden, self.nhidden)

        self.weight = Parameter(torch.FloatTensor(self.nhidden, self.nhidden))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.nhidden)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        seq_fea = torch.matmul(input, self.weight)
        output = torch.spmm(adj, seq_fea)
        return output



class CNNModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNNModel, self).__init__()

        self.convs = nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=5, stride=1, padding=2)
        self.fcs = nn.Linear(input_dim, output_dim)
        self.act_fn = nn.ReLU()

    def forward(self, x):
        pro_fea = torch.unsqueeze(x, 0).permute(0, 2, 1)
        layer_inner = self.convs(pro_fea)
        layer_inner = self.act_fn(layer_inner)

        layer_inner = nn.MaxPool1d(3, stride=1, padding=1)(layer_inner)
        layer_inner = torch.squeeze(layer_inner)

        layer_inner = torch.sum(layer_inner, dim=1)
        layer_inner = self.fcs(layer_inner)
        out_fea = nn.Sigmoid()(layer_inner)

        return out_fea


class predict_ec(nn.Module):
    def __init__(self, hidden_dim):
        super(predict_ec, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, 2*hidden_dim)
        self.fc2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1024)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        x = torch.mean(x, dim=0)
        x = torch.relu(self.fc4(x))
        return x


class GCN(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, dropout):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden))

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(1024, nhidden))
        self.fcs.append(nn.Linear(nlayers * nhidden, nhidden))
        self.fcs.append(nn.Linear(2*nhidden, nhidden))

        self.act_fn = nn.ReLU()
        self.dropout = dropout

    def forward(self, x, adj, evo_fea):
        _layers = []
        local_fea = self.act_fn(self.fcs[0](x))

        for i, con in enumerate(self.convs):
            local_fea = self.act_fn(con(local_fea, adj))
            _layers.append(local_fea)

        local_fea = self.act_fn(self.fcs[2](torch.cat(_layers, 1)))

        global_fea = F.dropout(evo_fea, self.dropout, training=self.training)
        global_fea = self.act_fn(self.fcs[1](global_fea))

        profeas = self.act_fn(self.fcs[-1](torch.cat([global_fea, local_fea], 1)))
        return profeas


class SCREEN(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout):
        super(SCREEN, self).__init__()

        self.gcn = GCN(nlayers=nlayers, nfeat=nfeat, nhidden=nhidden, dropout=dropout)
        self.criterion = nn.CrossEntropyLoss()

        self.projection = nn.Linear(nhidden, nhidden//2)
        self.projection1 = nn.Linear(nhidden//2, nfeat)
        self.projection2 = nn.Linear(nfeat, nclass)
        self.act_fn =nn.ReLU()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
        self.predict_ec = predict_ec(nhidden)

    def forward(self, x, adj, evo_fea):
        enz_feas = self.gcn(x.float(), adj, evo_fea)

        inner_layer = self.act_fn(self.projection1(self.act_fn(self.projection(enz_feas))))
        output = self.projection2(inner_layer)
        ec_output = self.predict_ec(enz_feas)
        return output, ec_output



