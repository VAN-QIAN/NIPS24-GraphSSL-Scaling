import torch
import os.path as osp
import os
import pandas as pd
import numpy as np
import datetime
from logging import getLogger
import torch_geometric.transforms as T
from collections import namedtuple, Counter
from torch.utils.data.sampler import SubsetRandomSampler
from libgptb.data.dataset import DGLDataset
#from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid, TUDataset
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected, degree

from libgptb.data.dataset.abstract_dataset import AbstractDataset
import importlib
from torch_geometric.loader import DataLoader

#from libgptb.data.
from copy import deepcopy
import pdb
def collate_fn(batch):
    # graphs = [x[0].add_self_loop() for x in batch]
    graphs = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    batch_g = dgl.batch(graphs)
    labels = torch.cat(labels, dim=0)
    return batch_g, labels
class TUDataset_MAE(AbstractDataset):
    
    def __init__(self, config):
        
        self.config = config
        self.datasetName = self.config.get('dataset', '')
        self.batch_size = self.config.get('batch_size', 64)
        self.train_ratio = self.config.get("train_ratio",0.8)
        self.valid_ratio = self.config.get("valid_ratio",0.1)
        self.test_ratio = self.config.get("test_ratio",0.1)
        #self.dataset=TUDataset(root="./data",name=self.datasetName)
        self.dataset = TUDataset(root="./data", name=self.datasetName)
    
        self._load_data()
        config["num_classes"]=self.num_classes
        
        self.split_ratio = self.config.get('ratio', 0.1)
        if self.split_ratio != 0:
            self.split_for_train(self.split_ratio)
            print("split generated")

    def _load_data(self):
        self.dataset = list(self.dataset)
        graph = self.dataset[0]
        deg4feat=False
        #dataset1 = TUDataset(self.datasetName)
       
        
        if graph.x == None:
            if graph.y and not deg4feat:
                print("Use node label as node features")
                feature_dim = 0
                for g in dataset:
                    feature_dim = max(feature_dim, int(g.y.max().item()))
            
                feature_dim += 1
                for i, g in enumerate(dataset):
                    node_label = g.y.view(-1)
                    feat = F.one_hot(node_label, num_classes=int(feature_dim)).float()
                    dataset[i].x = feat
            else:
                print("Using degree as node features")
                feature_dim = 0
                degrees = []
                for g in dataset:
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
                for i, g in enumerate(dataset):
                    degrees = degree(g.edge_index[0])
                    degrees[degrees > MAX_DEGREES] = MAX_DEGREES
                    degrees = torch.Tensor([int(x) for x in degrees.numpy().tolist()])
                    feat = F.one_hot(degrees.to(torch.long), num_classes=int(feature_dim)).float()
                    g.x = feat
                    dataset[i] = g

        else:
            print("******** Use `attr` as node features ********")
        feature_dim = int(graph.num_features)

        labels = torch.tensor([x.y for x in self.dataset])
    
        num_classes = torch.max(labels).item() + 1
        for i, g in enumerate(self.dataset):
            self.dataset[i].edge_index = remove_self_loops(self.dataset[i].edge_index)[0]
            self.dataset[i].edge_index = add_self_loops(self.dataset[i].edge_index)[0]
        #dataset = [(g, g.y) for g in dataset]

        print(f"******** # Num Graphs: {len(self.dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")
    #return dataset, (feature_dim, num_classes)
        self.num_features=feature_dim
        self.num_classes=num_classes


    def get_data(self):
        indices = torch.load("./split/{}.pt".format(self.datasetName))
        if self.split_ratio==1:
            train_size = int(len(self.dataset) * self.train_ratio)
            # valid_size = int(len(self.dataset) * self.valid_ratio)
            # test_size = int(len(self.dataset) * self.test_ratio)
            partial_size = min(int(self.split_ratio*train_size),train_size)
            
            train_set = [self.dataset[i] for i in indices[:partial_size]]
            # valid_set = [self.dataset[i] for i in indices[train_size: train_size + valid_size]]
            # test_set = [self.dataset[i] for i in indices[train_size + valid_size:]]
        else:
            train_path = f"./split/{self.datasetName}/{self.datasetName}_train{self.train_ratio}_{self.split_ratio}.pt"
            train_indices = torch.load(train_path)
            train_set = [self.dataset[i] for i in train_indices]
        train_idx = torch.arange(len(self.dataset))
        train_sampler = SubsetRandomSampler(train_idx)
        #dataset1= CustomDataset(self.dataset)
        dataloader = DataLoader(self.dataset, batch_size=self.config['batch_size'], pin_memory=True)
        dataloader_eval = DataLoader(train_set, batch_size=self.config['batch_size'], shuffle=False)

        return{"train":dataloader,"valid":dataloader_eval,"test":dataloader_eval,"full":dataloader}
        
    def split_for_train(self,ratio):
        """
        @parameter (float ratio): ratio of the dataset
        @return: return a dataloader with splited dataset
        """
        assert self.train_ratio + self.test_ratio + self.test_ratio <= 1
        seed = self.config.get("seed",0)

        split_file_path = "./split/{}.pt".format(self.datasetName)
        if os.path.exists(split_file_path):
            indices = torch.load("./split/{}.pt".format(self.datasetName))
        else:
            torch.manual_seed(self.config.get("seed",0))
            indices = torch.randperm(len(self.dataset))
            torch.save(indices,"./split/{}.pt".format(self.datasetName))

        print(f"split_for_train seed{seed}:{indices[0:10]}")
        train_size = int(len(self.dataset) * self.train_ratio)
        valid_size = int(len(self.dataset) * self.valid_ratio)
        test_size = int(len(self.dataset) * self.test_ratio)

        self.train = indices[:train_size]
        self.valid = indices[train_size: train_size + valid_size],
        self.test = indices[train_size + valid_size:]
        
        if not os.path.exists(f"./split/{self.datasetName}"):
                os.makedirs(f"./split/{self.datasetName}")
        cur_ratio = ratio
        while cur_ratio <= 1:
            cur_partial = min(int(cur_ratio*train_size),train_size)
            torch.save(indices[:cur_partial],f"./split/{self.datasetName}/{self.datasetName}_train{self.train_ratio}_{cur_ratio}.pt")
            cur_ratio = round(cur_ratio + ratio, 1)

        torch.save(self.train,f"./split/{self.datasetName}/{self.datasetName}_train{self.train_ratio}.pt")
        torch.save(self.valid,f"./split/{self.datasetName}/{self.datasetName}_valid{self.valid_ratio}.pt")
        torch.save(self.test,f"./split/{self.datasetName}/{self.datasetName}_test{self.test_ratio}.pt") 

    
    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"num_features":self.num_features}
