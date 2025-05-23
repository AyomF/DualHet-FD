# 运行时间: 16:48
import math
import torch
import dgl
import copy
import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
from utils import hinge_loss
from utils import *
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class Label_Propagation(Module):
    def __init__(self, input_dim, output_dim, head, si_adj, bi_adj,labels,relation_aware, etype, dropout, if_sum=False, bias=True):
        super().__init__()
        self.in_features = input_dim
        self.out_features = output_dim
        self.weight = Parameter(torch.FloatTensor(input_dim, output_dim))
        self.weight_bi = Parameter(torch.FloatTensor(input_dim, output_dim))
        self.w = Parameter(torch.FloatTensor(1))
        if bias:
            self.bias = Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.adjacency_mask = Parameter(si_adj.clone())
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        stdv_bi = 1. / math.sqrt(self.weight_bi.size(1))
        self.weight_bi.data.uniform_(-stdv_bi, stdv_bi)
        self.w.data.uniform_(0.5, 1)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, dataset, features, labels, si_adj, bi_adj):
        T = bi_adj.clone()
        T = T * self.adjacency_mask
        T = F.normalize(T, p=1, dim=1)
        indices = torch.nonzero(T, as_tuple=True)
        t = T[indices]
        y_hat = torch.mm(T, labels)
        return t,y_hat

class Aggregation(nn.Module):
    def __init__(self, input_dim, output_dim, head, si_adj, bi_adj,labels,relation_aware, etype, dropout, if_sum=False):
        super().__init__()
        self.etype = etype
        self.head = head
        self.hd = output_dim
        self.if_sum = if_sum
        self.relation_aware = relation_aware
        self.w_liner = nn.Linear(input_dim, output_dim*head)
        self.atten = nn.Linear(2*self.hd, 1)
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.Homo = Label_Propagation(input_dim, output_dim, head, si_adj, bi_adj,labels,relation_aware, etype, dropout, if_sum=False, bias=True)

    def forward(self, dataset, features, labels, si_adj, bi_adj):
        with dataset.local_scope():
            dataset.ndata['feat'] = features
            homo,y_hat = self.Homo(dataset, features, labels, si_adj, bi_adj)
            dataset.apply_edges(self.sign_edges, etype=self.etype)
#-
            features = self.w_liner(features)
            dataset.ndata['h'] = features
            dataset.update_all(message_func=self.message, reduce_func=self.reduce, etype=self.etype)
            out = dataset.ndata['out']
            return out,y_hat



    def message(self, edges):
        src = edges.src

        src_features = edges.data['sign'].view(-1,1)*src['h']

        src_features = src_features.view(-1, self.head, self.hd)
        z = torch.cat([src_features, edges.dst['h'].view(-1, self.head, self.hd)], dim=-1)
        alpha = self.atten(z)
        alpha = self.leakyrelu(alpha)
        return {'atten':alpha, 'sf':src_features}

    def reduce(self, nodes):
        alpha = nodes.mailbox['atten']
        sf = nodes.mailbox['sf']
        alpha = self.softmax(alpha)
        out = torch.sum(alpha*sf, dim=1)
        if not self.if_sum:
            out = out.view(-1, self.head*self.hd)
        else:
            out = out.sum(dim=-2)
        return {'out':out}
#-

    def sign_edges(self, edges):
        src = edges.src['feat']
        dst = edges.dst['feat']
        score = self.relation_aware(src, dst)
        return{'sign':score}

class HeterophilyLearning(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        #                 25          32
        super().__init__()
        self.d_liner = nn.Linear(input_dim, output_dim)
        self.f_liner = nn.Linear(3*output_dim, 1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

#-
    def forward(self, src, dst):
        src = self.d_liner(src)
        dst = self.d_liner(dst)
        diff = src-dst
        e_feats = torch.cat([src, dst, diff], dim=1)
        e_feats = self.dropout(e_feats)
        score = self.f_liner(e_feats).squeeze()
        score = self.tanh(score)
        return score
#-


class MRDualHFDNetLayer(nn.Module):
    def __init__(self, input_dim, output_dim, head, dataset, features, labels, dropout, if_sum = False):
        super().__init__()
        self.relation = copy.deepcopy(dataset.etypes)
        self.relation.remove('homo')
        self.n_relation = len(self.relation)

        if not if_sum:
            self.liner = nn.Linear(self.n_relation*output_dim*head, output_dim*head)
        else:
            self.liner = nn.Linear(self.n_relation*output_dim, output_dim)
        self.relation_aware = HeterophilyLearning(input_dim, output_dim*head, dropout)
        self.minelayers = nn.ModuleDict()
        self.dropout = nn.Dropout(dropout)
        for e in self.relation:
            adjacency_matrix = dataset.adjacency_matrix(etype=e).to_dense()
            #symmetric_matrix = (adjacency_matrix + adjacency_matrix.T) / 2
            #adjacency_matrix[torch.arange(adjacency_matrix.size(0)), torch.arange(adjacency_matrix.size(1))] = 1
            #normalized_matrix = adjacency_matrix / adjacency_matrix.sum(dim=1, keepdims=True)
            self.si_adj = adjacency_matrix.clone()
            self.bi_adj = adjacency_matrix.mm(adjacency_matrix)
            self.minelayers[e] =  Aggregation(input_dim, output_dim, head, self.si_adj, self.bi_adj, labels,self.relation_aware, e, dropout, if_sum = True)


    def forward(self, dataset, features, labels):
        hs = []
        for e in self.relation:
            he,y_hat = self.minelayers[e](dataset, features, labels, self.si_adj, self.bi_adj)

            hs.append(he)
        h = torch.cat(hs, dim=1)
        h = self.dropout(h)
        h = self.liner(h)
        return h, y_hat



    def loss(self, dataset, features, labels):
        with dataset.local_scope():
            dataset.ndata['feat'] = features
            agg_h, y_hat = self.forward(dataset, features, labels)
#-
            dataset.apply_edges(self.score_edges, etype='homo')
            edges_score = dataset.edges['homo'].data['score']
            edge_train_mask = dataset.edges['homo'].data['train_mask'].bool()
            edge_train_label = dataset.edges['homo'].data['label'][edge_train_mask]
            edge_train_pos = edge_train_label == 1
            edge_train_neg = edge_train_label == -1
            edge_train_pos_index = edge_train_pos.nonzero().flatten().detach().cpu().numpy()
            edge_train_neg_index = edge_train_neg.nonzero().flatten().detach().cpu().numpy()
            edge_train_pos_index = np.random.choice(edge_train_pos_index, size=len(edge_train_neg_index))
            index = np.concatenate([edge_train_pos_index, edge_train_neg_index])
            index.sort()
            edge_train_score = edges_score[edge_train_mask]
            # hinge loss
            edge_diff_loss = hinge_loss(edge_train_label[index], edge_train_score[index])

            train_mask = dataset.ndata['train_mask'].bool()
            train_h = agg_h[train_mask]
            train_label = dataset.ndata['label'][train_mask]
            train_pos = train_label == 1
            train_neg = train_label == 0
            train_pos_index = train_pos.nonzero().flatten().detach().cpu().numpy()
            train_neg_index = train_neg.nonzero().flatten().detach().cpu().numpy()
            train_neg_index = np.random.choice(train_neg_index, size=len(train_pos_index))
            node_index = np.concatenate([train_neg_index, train_pos_index])
            node_index.sort()
            pos_prototype = torch.mean(train_h[train_pos], dim=0).view(1, -1)
            neg_prototype = torch.mean(train_h[train_neg], dim=0).view(1, -1)
            train_h_loss = train_h[node_index]
            lp_loss = F.nll_loss(y_hat[node_index], labels[node_index])
            pos_prototypes = pos_prototype.expand(train_h_loss.shape)
            neg_prototypes = neg_prototype.expand(train_h_loss.shape)
            diff_pos = - F.pairwise_distance(train_h_loss, pos_prototypes)
            diff_neg = - F.pairwise_distance(train_h_loss, neg_prototypes)
            diff_pos = diff_pos.view(-1, 1)
            diff_neg = diff_neg.view(-1, 1)
            diff = torch.cat([diff_neg, diff_pos], dim=1)
            diff_loss = F.cross_entropy(diff, train_label[node_index])


            return agg_h, edge_diff_loss, diff_loss, lp_loss

    def score_edges(self, edges):
        src = edges.src['feat']
        dst = edges.dst['feat']
        score = self.relation_aware(src, dst)
        return {'score':score}
#-








class DualHFDNet(nn.Module):
    def __init__(self, args, dataset, features, labels):
        super().__init__()
        self.n_layers = args.n_layer
        self.input_dim = dataset.nodes['r'].data['feature'].shape[1]
        self.intra_dim = args.intra_dim
        self.n_class = args.n_class
        self.gamma1 = args.gamma1
        self.gamma2 = args.gamma2
        self.n_layer = args.n_layer
        self.mine_layers = nn.ModuleList()

        if args.n_layer == 1:
            self.mine_layers.append(MRDualHFDNetLayer(self.input_dim, self.n_class, args.head, dataset, features, labels, args.dropout, if_sum=True))
        else:
            self.mine_layers.append(MRDualHFDNetLayer(self.input_dim, self.intra_dim, args.head, dataset, features, labels, args.dropout))

            for _ in range(1, self.n_layer-1):
                self.mine_layers.append(MRDualHFDNetLayer(self.intra_dim*args.head, self.intra_dim, args.head, dataset, features, labels, args.dropout))
            self.mine_layers.append(MRDualHFDNetLayer(self.intra_dim*args.head, self.n_class, args.head, dataset, features, labels, args.dropout, if_sum=True))
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

#-
    def forward(self, dataset):
        feats = dataset.ndata['feature'].float()
        h,y_hat = self.mine_layers[0](dataset, feats)
        if self.n_layer > 1:
            h = self.relu(h)
            h = self.dropout(h)
            for i in range(1, len(self.mine_layers)-1):
                h,y_hat = self.mine_layers[i](dataset, h)
                h = self.relu(h)
                h = self.dropout(h)
            h,y_hat = self.mine_layers[-1](dataset, h)
        return h
#-






    def loss(self, dataset, features, labels):
        feats = dataset.ndata['feature'].float()
        train_mask = dataset.ndata['train_mask'].bool()
        train_label = dataset.ndata['label'][train_mask]
        train_pos = train_label == 1
        train_neg = train_label == 0

        pos_index = train_pos.nonzero().flatten().detach().cpu().numpy()
        neg_index = train_neg.nonzero().flatten().detach().cpu().numpy()

        # neg_index = np.random.choice(neg_index, size=len(pos_index), replace=False)
        neg_index = np.random.choice(neg_index, size=len(pos_index), replace=True)

        index = np.concatenate([pos_index, neg_index])
        index.sort()

        h, edge_loss, prototype_loss, lp_loss = self.mine_layers[0].loss(dataset, features, labels)
#-
        if self.n_layer > 1:
            h = self.relu(h)
            h = self.dropout(h)
            for i in range(1, len(self.mine_layers)-1):
                h, e_loss, p_loss = self.mine_layers[i].loss(dataset, h)
                h = self.relu(h)
                h = self.dropout(h)
                edge_loss += e_loss
                prototype_loss += p_loss
            h, e_loss, p_loss = self.mine_layers[-1].loss(dataset, h)
            edge_loss += e_loss
            prototype_loss += p_loss
        model_loss = F.cross_entropy(h[train_mask][index], train_label[index])
        loss = model_loss + self.gamma1*edge_loss + self.gamma2*prototype_loss
        return loss



