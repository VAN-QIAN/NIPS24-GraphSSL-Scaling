import torch
import os.path as osp
import os
import pandas as pd
import numpy as np
import datetime
from logging import getLogger
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from libgptb.data.dataset.abstract_dataset import AbstractDataset
import importlib
from torch_geometric.loader import DataLoader

class TUDataset(AbstractDataset):
    def __init__(self, config):
        self.config = config
        self.datasetName = self.config.get('dataset', '')
        self.batch_size = self.config.get('batch_size', 64)
        self.train_ratio = self.config.get("train_ratio",0.8)
        self.valid_ratio = self.config.get("valid_ratio",0.1)
        self.test_ratio = self.config.get("test_ratio",0.1)
        
        self._load_data()
        self.ratio = self.config.get('ratio', 0)
        config['num_feature']={
            "input_dim": max(self.dataset.num_features, 1)
        }
        
        
    def _load_data(self):
        device = torch.device('cuda')
        path = osp.join(os.getcwd(), 'raw_data')

        # orignal paper choices of datasets.
        #if self.datasetName in ['MUTAG', 'PTC_MR', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K']:
        if self.datasetName in ["MUTAG", "MCF-7", "MOLT-4","P388","ZINC_full","reddit_threads"]:   
            tu_dataset = getattr(importlib.import_module('torch_geometric.datasets'), 'TUDataset')
        self.dataset = tu_dataset(path, name=self.datasetName, transform=T.NormalizeFeatures())
        labels = torch.tensor([x.y for x in self.dataset])
    
        self.num_classes = torch.max(labels).item() + 1
        
        feature_dim = int(self.dataset[0].num_features)
        
    def get_data(self):
        
        assert self.train_ratio + self.test_ratio + self.test_ratio <= 1
        indices = torch.load("./split/{}.pt".format(self.datasetName))

        train_size = int(len(self.dataset) * self.train_ratio)
        valid_size = int(len(self.dataset) * self.valid_ratio)
        test_size = int(len(self.dataset) * self.test_ratio)
        partial_size = min(int(self.ratio*train_size),train_size)
        
        train_set = [self.dataset[i] for i in indices[:partial_size]]
        valid_set = [self.dataset[i] for i in indices[train_size: train_size + valid_size]]
        test_set = [self.dataset[i] for i in indices[train_size + valid_size:]]

        return {
        'train': DataLoader(train_set, batch_size=self.batch_size),
        'valid': DataLoader(valid_set, batch_size=self.batch_size),
        'test': DataLoader(test_set, batch_size=self.batch_size),
        'full': DataLoader(self.dataset, batch_size=self.batch_size)
        }

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
