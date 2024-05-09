import torch
import os.path as osp
import os
import pandas as pd
import numpy as np
import datetime
from collections import namedtuple, Counter
import torch.nn.functional as F
from logging import getLogger
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from libgptb.data.dataset.abstract_dataset import AbstractDataset
from torch_geometric.utils import degree,add_self_loops, remove_self_loops, to_undirected
import importlib
from torch_geometric.loader import DataLoader
class TUDataset_MAE(AbstractDataset):
    def __init__(self, config):
        self.config = config
        self.datasetName = self.config.get('dataset', '').upper()
        self.batch_size = self.config.get('batch_size', 64)
        self.train_ratio = self.config.get("train_ratio",0.8)
        self.valid_ratio = self.config.get("valid_ratio",0.1)
        self.test_ratio = self.config.get("test_ratio",0.1)
        if self.datasetName in ["MUTAG", "MCF-7", "MOLT-4","P388","ZINC_full","reddit_threads"]:   
            tu_dataset = getattr(importlib.import_module('torch_geometric.datasets'), 'TUDataset')
        self.dataset = tu_dataset(root="./data", name=self.datasetName)
        self.dataset.num_features=0
        #self._load_data()
        self.ratio = self.config.get('ratio', 0)
#
#    def _load_data(self):
#        device = torch.device('cuda')
#        path = osp.join(os.getcwd(), 'raw_data')
#
#        # orignal paper choices of datasets.
#        #if self.datasetName in ['MUTAG', 'PTC_MR', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K']:
#        if self.datasetName in ["MUTAG", "MCF-7", "MOLT-4","P388","ZINC_full","reddit_threads"]:   
#            tu_dataset = getattr(importlib.import_module('torch_geometric.datasets'), 'TUDataset')
#        self.dataset = tu_dataset(path, name=self.datasetName, transform=T.NormalizeFeatures())
    
    def get_data(self):
        deg4feat=False
        graph=self.dataset[0]
        if graph.x == None:
            if graph.y and not deg4feat:
                print("Use node label as node features")
                feature_dim = 0
            for g in self.dataset:
                feature_dim = max(feature_dim, int(g.y.max().item()))
            
            feature_dim += 1
            for i, g in enumerate(self.dataset):
                node_label = g.y.view(-1)
                feat = F.one_hot(node_label, num_classes=int(feature_dim)).float()
                self.dataset[i].x = feat
            else:
                print("Using degree as node features")
                feature_dim = 0
                degrees = []
                for g in self.dataset:
                    feature_dim = max(feature_dim, degree(g.edge_index[0]).max().item())
                    degrees.extend(degree(g.edge_index[0]).tolist())
                MAX_DEGREES = 400

                oversize = 0
                for d, n in Counter(degrees).items():
                    if d > MAX_DEGREES:
                        oversize += n
                # print(f"N > {MAX_DEGREES}, #NUM: {oversize}, ratio: {oversize/sum(degrees):.8f}")
                feature_dim = min(feature_dim, MAX_DEGREES)

                feature_dim += 1
                for i, g in enumerate(self.dataset):
                    degrees = degree(g.edge_index[0])
                    degrees[degrees > MAX_DEGREES] = MAX_DEGREES
                    degrees = torch.Tensor([int(x) for x in degrees.numpy().tolist()])
                    feat = F.one_hot(degrees.to(torch.long), num_classes=int(feature_dim)).float()
                    g.x = feat
                    self.dataset[i] = g

        else:
            print("******** Use `attr` as node features ********")
        feature_dim = int(graph.num_features)

        labels = torch.tensor([x.y for x in self.dataset])
    
        num_classes = torch.max(labels).item() + 1
        for i, g in enumerate(self.dataset):
            self.dataset[i].edge_index = remove_self_loops(self.dataset[i].edge_index)[0]
            self.dataset[i].edge_index = add_self_loops(self.dataset[i].edge_index)[0]
    #dataset = [(g, g.y) for g in dataset]
        self.dataset.num_features=feature_dim
        print(f"******** # Num Graphs: {len(self.dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")
        return self.dataset, (feature_dim, num_classes)
        

    def get_data_feature(self):
        """
        返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是点的个数，
        feature_dim是输入数据的维度，output_dim是模型输出的维度

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        
        return {
            "input_dim": max(self.dataset.num_features, 1)
        }


if __name__ == '__main__':
    for d in ['MCF-7', 'QM9']:   
        tu = TUDataset({"dataset":f"{d}"})
