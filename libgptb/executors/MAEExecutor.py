import os
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

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score

from libgptb.graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from libgptb.graphmae.datasets.data_util import load_graph_classification_dataset
from libgptb.graphmae.models import build_model
from torch.utils.tensorboard import SummaryWriter
from libgptb.executors.abstract_executor import AbstractExecutor
from libgptb.utils import get_evaluator, ensure_dir
from libgptb.evaluators import get_split, LREvaluator
from functools import partial
from libgptb.augmentors import EdgeRemovingDGL, FeatureMaskingDGL
class MAEExecutor(AbstractExecutor):
    def __init__(self, config, model, data_feature):
        self.exp_id = self.config.get('exp_id', None)

        self.cache_dir = './libgptb/cache/{}/model_cache'.format(self.exp_id)
        self.config=config
        self.evaluator=get_evaluator(config)
        self.model=build_model(config)
        self.data_feature=data_feature
        self.device = self.config.get('device', torch.device('cpu'))
        self.model=model.gnn.to(self.device)
        self.exp_id = self.config.get('exp_id', None)
        self.device = config.device if args.device >= 0 else "cpu"
        self.seeds = config.seeds
        self.dataset_name = config.dataset
        self.max_epoch = config.max_epoch
        self.max_epoch_f =config.max_epoch_f
        self.num_hidden = config.num_hidden
        self.num_layers = config.num_layers
        self.encoder_type = config.encoder
        self.decoder_type = config.decoder
        self.replace_rate = config.replace_rate

        self.optim_type = config.optimizer 
        self.loss_fn = config.loss_fn

        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.lr_f = config.lr_f
        self.weight_decay_f = config.weight_decay_f
        self.linear_prob = config.linear_prob
        self.load_model = config.load_model
        self.save_model = config.save_model
        self.logs = config.logging
        self.use_scheduler = config.scheduler
        self.pooler = config.pooling
        self.deg4feat = config.deg4feat
        self.batch_size = config.batch_size
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
        self._logger.info("Loaded model at {}".format(epoch))

    def pretrain(model, pooler, dataloaders, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob=True, logger=None):
        train_loader, eval_loader = dataloaders

        epoch_iter = tqdm(range(max_epoch))
        for epoch in epoch_iter:
            model.train()
            loss_list = []
            for batch in train_loader:
                batch_g = batch
                batch_g = batch_g.to(device)

                feat = batch_g.x
                model.train()
                loss, loss_dict = model(feat, batch_g.edge_index)
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_list.append(loss.item())
                if logger is not None:
                    loss_dict["lr"] = get_current_lr(optimizer)
                    logger.note(loss_dict, step=epoch)
            if scheduler is not None:
                scheduler.step()
        epoch_iter.set_description(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f}")
        
        return model
    def graph_classification_evaluation(self,model, pooler, dataloader, num_classes, lr_f, weight_decay_f, max_epoch_f, device, mute=False):
        model.eval()
        x_list = []
        y_list = []
        with torch.no_grad():
            for i, batch_g in enumerate(dataloader):
                batch_g = batch_g.to(device)
                feat = batch_g.x
                labels = batch_g.y.cpu()
                out = model.embed(feat, batch_g.edge_index)
                if pooler == "mean":
                    out = global_mean_pool(out, batch_g.batch)
                elif pooler == "max":
                    out = global_max_pool(out, batch_g.batch)
                elif pooler == "sum":
                    out = global_add_pool(out, batch_g.batch)
                else:
                    raise NotImplementedError

                y_list.append(labels.numpy())
                x_list.append(out.cpu().numpy())
        x = np.concatenate(x_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        test_f1, test_std = self.evaluate_graph_embeddings_using_svm(x, y)
        print(f"#Test_f1: {test_f1:.4f}±{test_std:.4f}")
        return test_f1


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
    def evaluate(self, dataloader):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')
        #for epoch_idx in [50-1, 100-1, 500-1, 1000-1, 10000-1]:
        for epoch_idx in [10-1,20-1,40-1,60-1,80-1,100-1]:
            self.load_model_with_epoch(epoch_idx)
            self.model.encoder_model.eval()
            x = []
            y = []
            for data in dataloader:
                data = data.to('cuda')
                if data.x is None:
                    num_nodes = data.batch.size(0)
                    data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
                _, _, g1, g2 = self.model.encoder_model(data.x, data.edge_index, data.batch)
                x.append(g1 + g2)
                y.append(data.y)
            x = torch.cat(x, dim=0)
            y = torch.cat(y, dim=0)

            split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1,dataset=self.config['dataset'])
            result = SVMEvaluator()(x, y, split)
            print(f'(E): Best test F1Mi={result["micro_f1"]:.4f}, F1Ma={result["macro_f1"]:.4f}')


            self._logger.info('Evaluate result is ' + json.dumps(result))
            filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + \
                        self.config['model'] + '_' + self.config['dataset']
            save_path = self.evaluate_res_dir
            with open(os.path.join(save_path, '{}.json'.format(filename)), 'w') as f:
                json.dump(result, f)
                self._logger.info('Evaluate result is saved at ' + os.path.join(save_path, '{}.json'.format(filename)))
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
        self._logger.info("num_batches:{}".format(num_batches))
        epoch_idx=0
        for epoch_idx in 1000:
            print(f"####### Running for epoch {epoch_idx}")
            #set_random_seed(seed)

            if self.logs:
                logger = TBLogger(name=f"{self.dataset_name}_loss_{self.loss_fn}_rpr_{self.replace_rate}_nh_{self.num_hidden}_nl_{self.num_layers}_lr_{self.lr}_mp_{self.max_epoch}_mpf_{self.max_epoch_f}_wd_{self.weight_decay}_wdf_{self.weight_decay_f}_{self.encoder_type}_{self.decoder_type}")
            else:
                logger = None

            self.model = build_model(self.config)
            self.model.to(self.device)
            optimizer = create_optimizer(self.optim_type, self.model, self.lr, self.weight_decay)

        
            
            
            self.model = self.pretrain(self.create_optimizermodel, self.pooler, (self.train_loader, self.eval_loader), optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob,  logger)
            self.model = self.model.cpu()
            if epoch_idx+1 in [10,20,40,60,80,100]:
                model_file_name = self.save_model_with_epoch(epoch_idx)
                self._logger.info('saving to {}'.format(model_file_name))

            
        
    
        return

    