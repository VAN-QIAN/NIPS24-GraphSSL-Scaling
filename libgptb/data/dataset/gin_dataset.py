import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
#from gnn import GNN
from libgptb.data.dataset.abstract_dataset import AbstractDataset
from tqdm import tqdm
import argparse
import time
import numpy as np
import sys
import random
### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator



class GINDataset(AbstractDataset):

    def __init__(self, config):
        self.config = config
        self.datasetName = self.config.get('dataset', '')
        self._load_data()

    def _load_data(self):

        dataset = PygGraphPropPredDataset(name = self.datasetName)

        if self.config.get("feature") == 'full':
            pass 
        elif self.config.get("feature") == 'simple':
            print('using simple feature')
            # only retain the top two node/edge features
            dataset.x = dataset.x[:,:2]
            dataset.edge_attr = dataset.edge_attr[:,:2]

        self.split_idx = dataset.get_idx_split()
        
        self.data = dataset

    def get_data(self):
        return self.data
    
    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        raise NotImplementedError("get_data_feature not implemented")