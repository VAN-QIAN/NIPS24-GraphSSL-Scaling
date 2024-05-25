import os
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import time
import json
from sklearn.utils import column_or_1d
import numpy as np
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from logging import getLogger
from torch import optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from libgptb.evaluators import get_split, SVMEvaluator,PyTorchEvaluator
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from torch.utils.tensorboard import SummaryWriter
from libgptb.executors.abstract_executor import AbstractExecutor
from libgptb.utils import get_evaluator, ensure_dir
from libgptb.evaluators import get_split, LREvaluator
from sklearn.svm import SVC

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


class MAEExecutor(AbstractExecutor):
    def __init__(self, config, model, data_feature):
        
        self.config=config
        self.exp_id = self.config.get('exp_id', None)
        self.cache_dir = './libgptb/cache/{}/model_cache'.format(self.exp_id)
        self.summary_writer_dir = './libgptb/cache/{}/'.format(self.exp_id)
        self.config=config
        self._logger = getLogger()
        
        self.data_feature=data_feature
        self.device = self.config.get('device', torch.device('cpu'))
        self.exp_id = self.config.get('exp_id', None)
        self.device = config.get('device',0)
        self.dataset_name = config['dataset']
        self.max_epoch = config['max_epoch']
        self.evaluate_res_dir = './libgptb/cache/{}/evaluate_cache'.format(self.exp_id)
        self.max_epoch_f =config['max_epoch_f']
        self.num_hidden = config['nhid']
        self.num_layers = config['num_layers']
        self.nclasses=self.config.get('num_classes', 2)
        self.encoder_type = config['encoder']
        self.decoder_type = config['decoder']
        self.optim_type = config['optimizer'] 
        self.loss_fn = config['loss_fn']
        self.learner = self.config.get('learner', 'adam')
        self.learning_rate = config['learning_rate']
        print("lr: {}".format(self.learning_rate))
        self.weight_decay = config['weight_decay']
        self.weight_decay_f = config['weight_decay_f']
        self.linear_prob = config['linear_prob']
        self.scheduler = config['scheduler']
        self.pooler = config['pooling']
        self.deg4feat = config['deg4feat']
        self.batch_size = config['batch_size']
        self.model=model
        self.downstream_task=config.get("downstream_task","original")
        self.downstream_ratio=self.config.get("downstream_ratio",0.1)
        self.load_best_epoch = self.config.get('load_best_epoch', False)
        self.patience = self.config.get('patience', 50)
        self.saved = self.config.get('saved_model', True)
        self.log_every = self.config.get('log_every', 1)
        self._writer = SummaryWriter(self.summary_writer_dir)
        self.lr_decay = self.config.get('lr_decay', True)
        self.lr_decay_ratio = self.config.get('lr_decay_ratio', 0.1)
        self.lr_scheduler_type = self.config.get('lr_scheduler', 'multisteplr')
        self.lr_scheduler = self._build_lr_scheduler()
        self.use_early_stop = self.config.get('use_early_stop', False)
        self._epoch_num = self.config.get('epoch', 0)
        if self._epoch_num > 0:
            self.load_model_with_epoch(self._epoch_num)
        #print(self.model)
    def _build_lr_scheduler(self):
        """
        根据全局参数`lr_scheduler`选择对应的lr_scheduler
        """
        if self.lr_decay:
            self._logger.info('You select `{}` lr_scheduler.'.format(self.lr_scheduler_type.lower()))
            if self.lr_scheduler_type.lower() == 'multisteplr':
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer, milestones=self.milestones, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'steplr':
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=self.step_size, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'exponentiallr':
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'cosineannealinglr':
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.lr_T_max, eta_min=self.lr_eta_min)
            elif self.lr_scheduler_type.lower() == 'lambdalr':
                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lr_lambda=self.lr_lambda)
            elif self.lr_scheduler_type.lower() == 'reducelronplateau':
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', patience=self.lr_patience,
                    factor=self.lr_decay_ratio, threshold=self.lr_threshold)
            else:
                self._logger.warning('Received unrecognized lr_scheduler, '
                                     'please check the parameter `lr_scheduler`.')
                lr_scheduler = None
        else:
            lr_scheduler = None
        return lr_scheduler
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



  

    

    def evaluate(self, test_dataloader):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        i=0
        self._logger.info('Start evaluating ...')
        #for epoch_idx in [50-1, 100-1, 500-1, 1000-1, 10000-1]:
        for epoch_idx in [10-1,20-1,40-1,60-1,80-1,100-1]:

            if epoch_idx+1 > self.max_epoch:
                break
            self.load_model_with_epoch(epoch_idx)
            losses = self._train_epoch(test_dataloader, epoch_idx)
            result2 = np.mean(losses)  
             
            self.model.eval()
            x_list = []
            y_list = []
            with torch.no_grad():
                for i, batch_g in enumerate(test_dataloader):
                   
                    batch_g = batch_g.to(self.device)
                    feat = batch_g.x
                    labels = batch_g.y.cpu()
                    out = self.model.embed(feat, batch_g.edge_index)
                    if self.pooler == "mean":
                        out = global_mean_pool(out, batch_g.batch)
                    elif self.pooler == "max":
                        out = global_max_pool(out, batch_g.batch)
                    elif self.pooler == "sum":
                        out = global_add_pool(out, batch_g.batch)
                    else:
                        raise NotImplementedError
                    
                    #print(labels.numpy())
                    #print(out.cpu().numpy)
                    y_list.append(labels)
                    x_list.append(out)
            x = torch.cat(x_list, dim=0)
            y = torch.cat(y_list, dim=0)
            split = get_split(num_samples=x.shape[0], train_ratio=0.8, test_ratio=0.1,dataset=self.config['dataset'])
            #result = SVMEvaluator(linear=True)(x, y, split)
            #test_f1, test_std = evaluate_graph_embeddings_using_svm(x, y)
            result=PyTorchEvaluator(n_features=x.shape[1],n_classes=self.nclasses)(x, y, split)
            self._logger.info('Evaluate result is ' + json.dumps(result))
            #self._logger.info('Evaluate result is macro' + (test_f1+test_std)+'   min_f1    '+(test_f1-test_std) )
            
            self._logger.info('Evaluate result2 is ' + json.dumps(result2))
            filename = 'epoch'+str(epoch_idx)+"_"+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + \
                            self.config['model'] + '_' + self.config['dataset']
            save_path = self.evaluate_res_dir
            file_path = os.path.join(save_path, '{}.json'.format(filename))
            if not os.path.exists(file_path):
                os.makedirs(save_path, exist_ok=True)
            print(filename)
            i+=1
            with open(os.path.join(save_path, '{}.json'.format(filename)), 'w') as f:
                json.dump(result, f)
                self._logger.info('Evaluate result is saved at ' + os.path.join(save_path, '{}.json'.format(filename)))
    def create_optimizer(self):
        """
        根据全局参数`learner`选择optimizer
        """
        self._logger.info('You select `{}` optimizer.'.format(self.learner.lower()))
        if self.learner.lower() == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                        momentum=self.lr_momentum, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate,
                                            eps=self.lr_epsilon, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate,
                                            alpha=self.lr_alpha, eps=self.lr_epsilon,
                                            momentum=self.lr_momentum, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sparse_adam':
            optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=self.learning_rate,
                                               eps=self.lr_epsilon, betas=self.lr_betas)
        else:
            self._logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = torch.optim.Adam(self.model.encoder_model.parameters(), lr=self.learning_rate,
                                         eps=self.lr_epsilon, weight_decay=self.weight_decay)
        return optimizer

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
        self.model.to(self.device)
        optimizer = self.create_optimizer()
        #print(optimizer)
        #print(self.config["num_classes"])
        self.optimizer=optimizer
        wait = 0
        best_epoch = 0
        train_time = []
        eval_time = []
        num_batches = len(train_dataloader)
        self._logger.info("num_batches:{}".format(num_batches))
        if self.config['exp_id']=='60997':
            return
        for epoch_idx in range(self.max_epoch):
            start_time = time.time()
            losses = self._train_epoch(train_dataloader, epoch_idx)
            t1 = time.time()
            train_time.append(t1 - start_time)
            self._writer.add_scalar('training loss', np.mean(losses), epoch_idx)
            self._logger.info("epoch complete!")

            self._logger.info("evaluating now!")
            t2 = time.time()
            val_loss = np.mean(losses) 
            end_time = time.time()
            eval_time.append(end_time - t2)

            if self.lr_scheduler is not None:
                if self.lr_scheduler_type.lower() == 'reducelronplateau':
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

            if (epoch_idx % self.log_every) == 0:
                log_lr = self.optimizer.param_groups[0]['lr']
                message = 'Epoch [{}/{}] train_loss: {:.4f}, lr: {:.6f}, {:.2f}s'.\
                    format(epoch_idx, self.max_epoch, np.mean(losses),  log_lr, (end_time - start_time))
                self._logger.info(message)
            """
            if epoch_idx+1 in [1] and self.config['dataset'] in ["ogbg-ppa"]:
                model_file_name = self.save_model_with_epoch(epoch_idx)
                self._logger.info('saving to {}'.format(model_file_name))
                break
            """
            #if epoch_idx+1 in [50, 100, 500, 1000, 10000]:
            if epoch_idx+1 in [10,20,40,60,80,100]:
                model_file_name = self.save_model_with_epoch(epoch_idx)
                self._logger.info('saving to {}'.format(model_file_name))

            if val_loss < min_val_loss:
                wait = 0
                if self.saved:
                    model_file_name = self.save_model_with_epoch(epoch_idx)
                    self._logger.info('Val loss decrease from {:.4f} to {:.4f}, '
                                      'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss
                best_epoch = epoch_idx
            else:
                wait += 1
                if wait == self.patience and self.use_early_stop:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_idx)
                    break
        if len(train_time) > 0:
            self._logger.info('Trained totally {} epochs, average train time is {:.3f}s, '
                              'average eval time is {:.3f}s'.
                              format(len(train_time), sum(train_time) / len(train_time),
                                     sum(eval_time) / len(eval_time)))
        if self.load_best_epoch:
            self.load_model_with_epoch(best_epoch)
        return min_val_loss

    def _train_epoch(self, train_dataloader,epoch_idx,loss_func=None,train=True):
        loss_all = 0
        num_graph=len(train_dataloader)
        
        if train:
            self.model.train()
        else:
            self.model.eval()
        for batch in train_dataloader:
            #print(batch)
            batch_g = batch
            batch_g = batch_g.to(self.device)

            feat = batch_g.x
            self.model.train()
            loss, loss_dict = self.model(feat, batch_g.edge_index)
            
            self.optimizer.zero_grad()
            loss_all+=loss.item()
            if train:
                loss.backward()
                self.optimizer.step()
          
        return loss_all 
