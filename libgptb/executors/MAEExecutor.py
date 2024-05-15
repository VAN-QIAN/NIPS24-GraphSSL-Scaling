import os
import dgl
from dgl.data import (
    load_data, 
    TUDataset, 
    CoraGraphDataset, 
    CiteseerGraphDataset, 
    PubmedGraphDataset
)
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader
import torch.nn.functional as F
import time
import json
import numpy as np
import datetime
import torch
from tqdm import tqdm
from logging import getLogger
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import logging
from tqdm import tqdm
import numpy as np
import torch
from torch import optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from libgptb.evaluators import get_split, SVMEvaluator
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from libgptb.graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from libgptb.graphmae.datasets.data_util import load_graph_classification_dataset
from libgptb.graphmae.models import build_model
from torch.utils.tensorboard import SummaryWriter
from libgptb.executors.abstract_executor import AbstractExecutor
from libgptb.utils import get_evaluator, ensure_dir
from libgptb.evaluators import get_split, LREvaluator
from functools import partial
from libgptb.augmentors import EdgeRemovingDGL, FeatureMaskingDGL
from logging import getLogger
def evaluate_graph_embeddings_using_svm(embeddings, labels):
    result = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    for train_index, test_index in kf.split(embeddings, labels):
        x_train = embeddings[train_index]
        x_test = embeddings[test_index]
        y_train = labels[train_index]
        y_test = labels[test_index]
        params = {"C": [1e-3, 1e-2, 1e-1, 1, 10]}
        svc = SVC(random_state=42)
        clf = GridSearchCV(svc, params)
        clf.fit(x_train, y_train)

        preds = clf.predict(x_test)
        f1 = f1_score(y_test, preds, average="micro")
        result.append(f1)
    test_f1 = np.mean(result)
    test_std = np.std(result)

    return test_f1, test_std
def collate_fn(batch):
    # graphs = [x[0].add_self_loop() for x in batch]
    graphs = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    batch_g = dgl.batch(graphs)
    labels = torch.cat(labels, dim=0)
    return batch_g, labels
def load_graph_classification_dataset(dataset_name, deg4feat=False):
    dataset_name = dataset_name.upper()
    dataset = TUDataset(dataset_name)
    graph, _ = dataset[0]

    if "attr" not in graph.ndata:
        if "node_labels" in graph.ndata and not deg4feat:
            print("Use node label as node features")
            feature_dim = 0
            for g, _ in dataset:
                feature_dim = max(feature_dim, g.ndata["node_labels"].max().item())
            
            feature_dim += 1
            for g, l in dataset:
                node_label = g.ndata["node_labels"].view(-1)
                feat = F.one_hot(node_label, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
        else:
            print("Using degree as node features")
            feature_dim = 0
            degrees = []
            for g, _ in dataset:
                feature_dim = max(feature_dim, g.in_degrees().max().item())
                degrees.extend(g.in_degrees().tolist())
            MAX_DEGREES = 400

            oversize = 0
            for d, n in Counter(degrees).items():
                if d > MAX_DEGREES:
                    oversize += n
            # print(f"N > {MAX_DEGREES}, #NUM: {oversize}, ratio: {oversize/sum(degrees):.8f}")
            feature_dim = min(feature_dim, MAX_DEGREES)

            feature_dim += 1
            for g, l in dataset:
                degrees = g.in_degrees()
                degrees[degrees > MAX_DEGREES] = MAX_DEGREES
                
                feat = F.one_hot(degrees, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
    else:
        print("******** Use `attr` as node features ********")
        feature_dim = graph.ndata["attr"].shape[1]

    labels = torch.tensor([x[1] for x in dataset])
    
    num_classes = torch.max(labels).item() + 1
    dataset = [(g.remove_self_loop().add_self_loop(), y) for g, y in dataset]

    print(f"******** # Num Graphs: {len(dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")

    return dataset, (feature_dim, num_classes)
class MAEExecutor(AbstractExecutor):
    def __init__(self, config, model, data_feature):
        
        print("enter initialize")
        self.config=config
        self.exp_id = self.config.get('exp_id', None)
        print("first")
        self.cache_dir = './libgptb/cache/{}/model_cache'.format(self.exp_id)
        self.config=config
        self._logger = getLogger()
        print("next2")
        print(config['num_heads'])


        
        self.evaluator=get_evaluator(config)
        print("next3")
        #self.model=build_model(config)
        #self.model.enco
        print("next")
        self.data_feature=data_feature
        self.device = self.config.get('device', torch.device('cpu'))
        #self.model=model.gnn.to(self.device)
        self.exp_id = self.config.get('exp_id', None)
        self.device = config.get('device',0)
        #self.seeds = config['seeds']
        self.dataset_name = config['dataset']
        self.max_epoch = config['max_epoch']
        self.evaluate_res_dir = './libgptb/cache/{}/evaluate_cache'.format(self.exp_id)
        self.max_epoch_f =config['max_epoch_f']
        self.num_hidden = config['num_hidden']
        self.num_layers = config['num_layers']
        self.encoder_type = config['encoder']
        self.decoder_type = config['decoder']
        self.replace_rate = config['replace_rate']

        self.optim_type = config['optimizer'] 
        self.loss_fn = config['loss_fn']

        self.lr = config['lr']
        self.weight_decay = config['weight_decay']
        self.lr_f = config['lr_f']
        self.weight_decay_f = config['weight_decay_f']
        self.linear_prob = config['linear_prob']
        #self.scheduler
        self.logs = config['logging']
        self.scheduler = config['scheduler']
        self.pooler = config['pooling']
        self.deg4feat = config['deg4feat']
        self.batch_size = config['batch_size']
        #self.model=build_model(config)
        #self._logger.info(self.model)
    def save_model(self, cache_name):
        """
        将当前的模型保存到文件

        Args:
            cache_name(str): 保存的文件名
        """
        ensure_dir(self.cache_dir)
        self._logger.info("Saved model at " + cache_name)
        torch.save((self.model.state_dict(), self.optimizer.state_dict()), cache_name)

    def load_model(self, cache_name):
        """
        加载对应模型的 cache

        Args:
            cache_name(str): 保存的文件名
        """
        self._logger.info("Loaded model at " + cache_name)
        model_state, optimizer_state = torch.load(cache_name)
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)
    def save_model_with_epoch(self, epoch):
        """
        保存某个epoch的模型

        Args:
            epoch(int): 轮数
        """
        
        ensure_dir(self.cache_dir)
        config = dict()
        config['model_state_dict'] = self.model.state_dict()
        config['optimizer_state_dict'] = self.optimizer.state_dict()
        config['epoch'] = epoch
        model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '_epoch%d.tar' % epoch
        torch.save(config, model_path)
        self._logger.info("Saved model at {}".format(epoch))
        
        return model_path

    def load_model_with_epoch(self, epoch):
        """
        加载某个epoch的模型

        Args:
            epoch(int): 轮数
        """
        model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '_epoch%d.tar' % epoch
        assert os.path.exists(model_path), 'Weights at epoch %d not found' % epoch
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #self.pooler.load_state_dict(check)
        self._logger.info("Loaded model at {}".format(epoch))

    def pretrain(self, model, pooler, dataloaders, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob=True, logger=None):
        train_loader, eval_loader = dataloaders

        epoch_iter = tqdm(range(max_epoch))
        for epoch in epoch_iter:
            model.train()
            loss_list = []
            for batch in train_loader:
                batch_g, _ = batch
                batch_g = batch_g.to(device)

                feat = batch_g.ndata["attr"]
                model.train()
                loss, loss_dict = model(batch_g, feat)
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_list.append(loss.item())
                if self._logger is not None:
                    loss_dict["lr"] = get_current_lr(optimizer)
                    #self._logger.info("loss_dict:{} epoch:{}".format(loss_dict,epoch))
            #if self.scheduler is not None:
            #    self.scheduler.step()
            epoch_iter.set_description(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f}")

        return model

    def graph_classification_evaluation(self,model, pooler, dataloader, num_classes, lr_f, weight_decay_f, max_epoch_f, device, mute=False):
        model.eval()
        x_list = []
        y_list = []
        if self.config['pooling'] == "mean":
            print("find mean")
            pooler1 = AvgPooling()
        elif self.config['pooling'] == "max":
            pooler1 = MaxPooling()
        elif self.config['pooling'] == "sum":
            pooler1 = SumPooling()
        else:
            raise NotImplementedError
        with torch.no_grad():
            for i, (batch_g,labels) in enumerate(dataloader):
                batch_g = batch_g.to(device)
                feat = batch_g.ndata["attr"]
                out = model.embed(batch_g, feat)
                out = pooler1(batch_g, out)

                y_list.append(labels.numpy())
                x_list.append(out.cpu().numpy())

                #y_list.append(labels.numpy())
                #x_list.append(out.cpu().numpy())
        print(x_list)
        x = np.concatenate(x_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        test_f1, test_std = self.evaluate_graph_embeddings_using_svm(x, y)
        print(f"#Test_f1: {test_f1:.4f}±{test_std:.4f}")
        return test_f1

    def evaluate_graph_embeddings_using_svm(self,embeddings, labels):
        result = []
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

        for train_index, test_index in kf.split(embeddings, labels):
            x_train = embeddings[train_index]
            x_test = embeddings[test_index]
            y_train = labels[train_index]
            y_test = labels[test_index]
            params = {"C": [1e-3, 1e-2, 1e-1, 1, 10]}
            svc = SVC(random_state=42)
            clf = GridSearchCV(svc, params)
            clf.fit(x_train, y_train)

            preds = clf.predict(x_test)
            f1 = f1_score(y_test, preds, average="micro")
            result.append(f1)
        test_f1 = np.mean(result)
        test_std = np.std(result)

        return test_f1, test_std

    def evaluate(self, dataloader):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        acc_list = []
        self._logger.info('Start evaluating ...')
        #for epoch_idx in [50-1, 100-1, 500-1, 1000-1, 10000-1]:
        for epoch_idx in [10-1,20-1,40-1,60-1,80-1,100-1]:
            self.load_model_with_epoch(epoch_idx)
            self.model = self.model.to(self.device)
            self.model.eval()
            test_f1 = self.graph_classification_evaluation(self.model, self.pooler, self.train_loader, self.num_layers, self.lr_f, self.weight_decay_f, self.max_epoch_f, self.device, mute=False)
            acc_list.append(test_f1)

            inal_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
            self._logger.info('inal_acc is ' + json.dumps(inal_acc))
            self._logger.info('final_acc_std is ' + json.dumps(final_acc_std))
            #filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + \
            #            self.config['model'] + '_' + self.config['dataset']
            #save_path = self.evaluate_res_dir
            #with open(os.path.join(save_path, '{}.json'.format(filename)), 'w') as f:
            #    json.dump(result, f)
            #    self._logger.info('Evaluate result is saved at ' + os.path.join(save_path, '{}.json'.format(filename)))
    def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
        opt_lower = opt.lower()

        parameters = model.parameters()
        opt_args = dict(lr=lr, weight_decay=weight_decay)

        opt_split = opt_lower.split("_")
        opt_lower = opt_split[-1]
        if opt_lower == "adam":
            optimizer = optim.Adam(parameters, **opt_args)
        elif opt_lower == "adamw":
            optimizer = optim.AdamW(parameters, **opt_args)
        elif opt_lower == "adadelta":
            optimizer = optim.Adadelta(parameters, **opt_args)
        elif opt_lower == "radam":
            optimizer = optim.RAdam(parameters, **opt_args)
        elif opt_lower == "sgd":
            opt_args["momentum"] = 0.9
            return optim.SGD(parameters, **opt_args)
        else:
            assert False and "Invalid optimizer"

        return optimizer
    def train(self, train_dataloader, eval_dataloader):
        """
        use data to train model with config

        Args:
            train_dataloader(torch.Dataloader): Dataloader
            eval_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = 0
        train_time = []
        eval_time = []
        num_batches = len(train_dataloader)
        graphs, (num_features, num_classes) = load_graph_classification_dataset(self.dataset_name, deg4feat=self.deg4feat)


        train_idx = torch.arange(len(graphs))
        train_sampler = SubsetRandomSampler(train_idx)
    
        self.train_loader = GraphDataLoader(graphs, sampler=train_sampler, collate_fn=collate_fn, batch_size=self.config['batch_size'], pin_memory=True)
        self.eval_loader = GraphDataLoader(graphs, collate_fn=collate_fn, batch_size=self.config['batch_size'], shuffle=False)
        self._logger.info("num_batches:{}".format(num_batches))
        epoch_idx=0
        for epoch_idx in range(100):
            print(f"####### Running for epoch {epoch_idx}")
            #set_random_seed(seed)

            if self.logs:
                logger = TBLogger(name=f"{self.dataset_name}_loss_{self.loss_fn}_rpr_{self.replace_rate}_nh_{self.num_hidden}_nl_{self.num_layers}_lr_{self.lr}_mp_{self.max_epoch}_mpf_{self.max_epoch_f}_wd_{self.weight_decay}_wdf_{self.weight_decay_f}_{self.encoder_type}_{self.decoder_type}")
            else:
                logger = None

            self.model = build_model(self.config)
            self.model.to(self.device)
            optimizer = create_optimizer(self.optim_type, self.model, self.lr, self.weight_decay)

            self.optimizer=optimizer

            
            self.model = self.pretrain(self.model, self.pooler, (self.train_loader, self.eval_loader), optimizer, self.max_epoch, self.device, self.scheduler, self.num_layers, self.lr_f, self.weight_decay_f, self.max_epoch_f, self.linear_prob,  self._logger)
            self.model = self.model.cpu()
            if epoch_idx+1 in [10,20,40,60,80,100]:
                model_file_name = self.save_model_with_epoch(epoch_idx)
                self._logger.info('saving to {}'.format(model_file_name))

            
        
    
        return

    