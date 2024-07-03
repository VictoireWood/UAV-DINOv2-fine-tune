import pytorch_lightning as pl
import torch
from torch.optim import lr_scheduler, optimizer

import utils
from models import helper


class VPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
        #---- Backbone
        backbone_arch='resnet50',         # NOTE - 原始
        backbone_config={},
        # backbone_arch='dinov2_vitb14',    # NOTE - 邵星雨
        # backbone_config={},
        
        #---- Aggregator
        agg_arch='ConvAP',                # NOTE - 原始
        agg_config={},
        # agg_arch='SALAD',                 # NOTE - 邵星雨
        # agg_config={},
        
        #---- Train hyperparameters
        lr=0.03,        # NOTE - 应该是learning rate
        optimizer='sgd',
        weight_decay=1e-3,
        momentum=0.9,
        lr_sched='linear',      # NOTE - 根据epoch训练次数来调整学习率（learning rate）的方法
        lr_sched_args = {
            'start_factor': 1,
            'end_factor': 0.2,
            'total_iters': 4000,
        },
        
        #----- Loss
        loss_name='MultiSimilarityLoss', 
        miner_name='MultiSimilarityMiner', 
        miner_margin=0.1,
        faiss_gpu=False
    ):
        # NOTE - 函数输入到这里结束
        super().__init__() # NOTE - 调用父类pl.LightningModule的初始化函数，是pl常规定义方法中必须有的第一步

        # Backbone
        self.encoder_arch = backbone_arch
        self.backbone_config = backbone_config
        
        # Aggregator
        self.agg_arch = agg_arch
        self.agg_config = agg_config

        # Train hyperparameters
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr_sched = lr_sched
        self.lr_sched_args = lr_sched_args

        # Loss
        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin
        
        self.save_hyperparameters() # write hyperparams into a file
        
        self.loss_fn = utils.get_loss(loss_name)    # NOTE - 参考<https://kevinmusgrave.github.io/pytorch-metric-learning/losses/>
        self.miner = utils.get_miner(miner_name, miner_margin)  # NOTE - 参考<https://kevinmusgrave.github.io/pytorch-metric-learning/miners/>
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 

        self.faiss_gpu = faiss_gpu
        
        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, backbone_config) # NOTE - 获取主网络
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)       # NOTE - 获取聚类方法（聚合层）

        # For validation in Lightning v2.0.0
        self.val_outputs = []   # NOTE - VPR模型的验证输出，一个列表，存储每个验证集的输出结果
        
    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)    # 这里得到的是feature map，后面SALAD是NetVLAD层的聚类，不知道单应性矩阵的聚类过程应该怎么写？
        x = self.aggregator(x)
        return x
    
    # configure the optimizer 
    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay, 
                momentum=self.momentum
            )
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        elif self.optimizer.lower() == 'adam':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')
        

        if self.lr_sched.lower() == 'multistep':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_sched_args['milestones'], gamma=self.lr_sched_args['gamma'])
        elif self.lr_sched.lower() == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.lr_sched_args['T_max'])
        elif self.lr_sched.lower() == 'linear':
            scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.lr_sched_args['start_factor'],
                end_factor=self.lr_sched_args['end_factor'],
                total_iters=self.lr_sched_args['total_iters']
            )

        return [optimizer], [scheduler]
    
    # configure the optizer step, takes into account the warmup stage
    def optimizer_step(self,  epoch, batch_idx, optimizer, optimizer_closure):
        # warm up lr
        optimizer.step(closure=optimizer_closure)   # NOTE - 执行单个优化步骤(参数更新)。Performs a single optimization step (parameter update).
        self.lr_schedulers().step()                 # NOTE - 参考<https://lightning.ai/docs/pytorch/stable/common/optimization.html>
        
    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        # we mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)
            
            # calculate the % of trivial pairs/triplets 
            # which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined/nb_samples)

        else: # no online mining
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple: 
                # somes losses do the online mining inside (they don't need a miner objet), 
                # so they return the loss and the batch accuracy
                # for example, if you are developping a new loss function, you might be better
                # doing the online mining strategy inside the forward function of the loss class, 
                # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
                loss, batch_acc = loss

        # keep accuracy of every batch and later reset it at epoch start
        self.batch_acc.append(batch_acc)
        # log it
        self.log('b_acc', sum(self.batch_acc) /
                len(self.batch_acc), prog_bar=True, logger=True)
        return loss
    
    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        places, labels = batch
        
        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places
        BS, N, ch, h, w = places.shape      # NOTE  - 这里需要看一下这个数据是怎么加载的？
        
        # reshape places and labels
        images = places.view(BS*N, ch, h, w)
        labels = labels.view(-1)

        # Feed forward the batch to the model
        descriptors = self(images) # Here we are calling the method forward that we defined above

        if torch.isnan(descriptors).any():
            raise ValueError('NaNs in descriptors')

        loss = self.loss_function(descriptors, labels) # Call the loss_function we defined above
        
        self.log('loss', loss.item(), logger=True, prog_bar=True)
        return {'loss': loss}
    
    def on_train_epoch_end(self):
        # we empty the batch_acc list for next epoch
        self.batch_acc = []

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        descriptors = self(places)
        self.val_outputs[dataloader_idx].append(descriptors.detach().cpu())     # NOTE - 这里的dataloader好像是用来加载数据集的，将每个验证集的结果放在val_outputs列表中
        return descriptors.detach().cpu()
        # NOTE - 官方文档说validation_step返回值中必须有loss tensor，不清楚为什么这个descriptor是loss。
        # NOTE - validation step的返回值应该取决于最后在main.py中调用ModelCheckpoint的时候，以什么为标准作为保存模型的依据。
        # 在这里是依据验证集的第一个结果的召回率R@1为依据，选结果排在前三的模型保存下来。
    
    # NOTE - Called in the validation loop at the very beginning of the epoch.
    def on_validation_epoch_start(self):
        # reset the outputs list
        # 形成空列表的列表，空列表个数和val_datasets长度相同，相当于初始化val_outputs
        self.val_outputs = [[] for _ in range(len(self.trainer.datamodule.val_datasets))]   # NOTE - 对应GSVCitiesDataModule.setup中的val_datasets
        # The line self.val_outputs = [[] for _ in range(len(self.trainer.datamodule.val_datasets))] is creating a list of empty lists. The length of the list is determined by the number of validation datasets (self.trainer.datamodule.val_datasets).
        # NOTE - self.trainer会调用Trainer，<https://lightning.ai/docs/pytorch/stable/common/trainer.html>
        # <https://adaning.github.io/posts/34610.html>在pl.LightningModule里会添加Trainer的Hook, 调用self.trainer就能够获得它身上的属性
    
    def on_validation_epoch_end(self):
        """this return descriptors in their order
        depending on how the validation dataset is implemented 
        for this project (MSLS val, Pittburg val), it is always references then queries
        [R1, R2, ..., Rn, Q1, Q2, ...]
        """
        val_step_outputs = self.val_outputs

        dm = self.trainer.datamodule
        # NOTE - Trainer.fit()参数datamodule是LightningDataModule(在dataloader文件里定义的)的实例<https://blog.csdn.net/qq_27135095/article/details/122654805>
        # 根据main.py，dm是GSVCitiesDataModule的实例
        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list (Pytorch Lightning does not do it presently)
        if len(dm.val_datasets)==1: # we need to put the outputs in a list
            val_step_outputs = [val_step_outputs]
        
        # NOTE - 这里需要改
        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            # i是各个(val_set_name, val_dataset)的索引（从0开始）
            # 默认是['pitts30k_val', 'msls_val']这两个val_dataset
            feats = torch.concat(val_step_outputs[i], dim=0)
            
            if 'pitts' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.dbStruct.numDb # Pittsburgh数据集database的图片数量(相对的，numQ是query的数量)
                positives = val_dataset.getPositives()  # 按拍摄点位最近邻搜索找到Query中的正样本
            elif 'msls' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            else:
                print(f'Please implement validation_epoch_end for {val_set_name}')
                raise NotImplemented

            r_list = feats[ : num_references]
            q_list = feats[num_references : ]
            pitts_dict = utils.get_validation_recalls(
                r_list=r_list,  # reference也就是database列表
                q_list=q_list,  # query列表
                k_values=[1, 5, 10, 15, 20, 50, 100],   # 召回多少个结果
                gt=positives,           # 真值（也就是根据最近邻搜索得到的正样本）
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu
            )
            del r_list, q_list, feats, num_references, positives

            self.log(f'{val_set_name}/R1', pitts_dict[1], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5', pitts_dict[5], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10', pitts_dict[10], prog_bar=False, logger=True)
        print('\n\n')

        # reset the outputs list
        self.val_outputs = []