import pytorch_lightning as pl
import torch
from torch.optim import lr_scheduler, optimizer

import utils
from models import helper

from GenerateDataset import CITIES, TRAIN_CITIES, VAL_CITIES


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
        # backbone_arch='dinov2_vitb14',    # NOTE = 邵星雨
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
        # NOTE - 没有理解这个跟踪是什么意思？ 

        self.faiss_gpu = faiss_gpu
        
        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, backbone_config) # NOTE - 获取主网络
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)       # NOTE - 获取聚类方法（聚合层）

        # For validation in Lightning v2.0.0
        self.val_outputs = []

    # NOTE - VPR的前向传播，也就是网络从头到尾的输入到输出的过程
    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x

    # configure the optimizer 
    # NOTE - 优化器主要用在模型训练阶段，用于更新模型中可学习的参数。
    # 选择要在优化中使用的优化器和学习率调度器。通常你需要一个。但在gan或类似的情况下你可能有多个。使用多个优化器的优化仅在手动优化模式下有效。
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
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
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
            batch_acc = 1.0 - (nb_mined/nb_samples) # NOTE - 这里不太理解

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
        
        # NOTE -logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log('loss', loss.item(), logger=True, prog_bar=True)
        return {'loss': loss}   # NOTE - training_step函数必须返回loss

    def on_train_epoch_end(self):
        # we empty the batch_acc list for next epoch
        self.batch_acc = []

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    # NOTE - 对验证集中的单个批数据进行操作。在这一步中，您可能会生成示例或计算任何感兴趣的内容，例如准确性。
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        # SECTION - 原始
        # NOTE - 官方文档示范：input, target = batch
        # places, _ = batch
        # descriptors = self(places)
        # self.val_outputs[dataloader_idx].append(descriptors.detach().cpu())     # NOTE - 这里的dataloader好像是用来加载数据集的
        # return descriptors.detach().cpu()
        # !SECTION

        # NOTE - 这里由于我在加载val数据集的时候使用了和train数据集一样的结构和加载方式，这里也需要仿照training_step写
        places, _ = batch
        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places
        
        descriptors = self(places)
        self.val_outputs[dataloader_idx].append(descriptors.detach().cpu())     # NOTE - 这里的dataloader好像是用来加载数据集的，将每个验证集的结果放在val_outputs列表中
        return descriptors.detach().cpu()

        # Feed forward the batch to the model
    def on_validation_epoch_start(self):
        # reset the outputs list
        # 形成空列表的列表，空列表个数和val_datasets长度相同，相当于初始化val_outputs
        self.val_outputs = [[] for _ in range(len(self.trainer.datamodule.val_datasets))]   # NOTE - 对应GSVCitiesDataModule.setup中的val_datasets

    def on_validation_epoch_end(self):
        # we empty the batch_acc list for next epoch
        val_step_outputs = self.val_outputs
        dm = self.trainer.datamodule
        # NOTE - Trainer.fit()参数datamodule是LightningDataModule(在dataloader文件里定义的)的实例<https://blog.csdn.net/qq_27135095/article/details/122654805>
        # 根据main.py，dm是GSVCitiesDataModule的实例
        if len(dm.val_datasets) == 1:
            val_step_outputs = [val_step_outputs]
        
        # recalls_rate = []
        recall_1st_rate = 0

        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            feats = torch.concat(val_step_outputs[i], dim=0)
            # 对应train cities
            num_references = val_dataset.dataset_info.numDb
            positives = val_dataset.getPositives()
        
            db_list = feats[ : num_references]
            q_list = feats[num_references : ]
            recalls_dict = utils.get_validation_recalls(
                r_list=db_list,     # reference也就是database列表
                q_list=q_list,      # query列表
                k_values=[1, 2],    # 召回多少个结果
                gt=positives,           # 真值（也就是根据最近邻搜索得到的正样本）
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu
            )
            del db_list, q_list, feats, num_references, positives

            self.log(f'{val_set_name}/R1', recalls_dict[1], prog_bar=False, logger=True)    # 这里的命名对应main.py中的ModelCheckpoint的monitor
            self.log(f'{val_set_name}/R2', recalls_dict[2], prog_bar=False, logger=True)
            # NOTE - 参考<https://lightning.ai/docs/pytorch/latest/extensions/logging.html#automatic-logging>

            # recalls_rate.append(recalls_dict[1], recalls_dict[2])
            recall_1st_rate += recalls_dict[1]

        print('\n\n')

        # 我甚至可以给所有val_dataset的R@1召回率求个平均
        ave_recall_1st_rate = recall_1st_rate / len(dm.val_datasets)
        self.log('Average_R@1', ave_recall_1st_rate, prog_bar=False, logger=True)


        # reset the outputs list
        self.val_outputs = []

    # FIXME - 原始程序没看懂，这里是怎么召回的


        