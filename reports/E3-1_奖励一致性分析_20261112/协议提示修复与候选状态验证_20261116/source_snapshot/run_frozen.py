import os
import copy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BaseFinetuning
from vilt.config import ex
from vilt.modules import ViLTransformerSS
from vilt.datamodules.multitask_datamodule import MTDataModule
from torch.optim.optimizer import Optimizer
import warnings
# from model.model import LayerNorm
from torch.nn.modules import LayerNorm, Linear


class MyBackboneFinetuning(BaseFinetuning):
    """
    PyTorch Lightning 回调：在训练开始前冻结 ViLT 的整个骨干网络（backbone），
    仅让任务相关的头部参数参与训练。

    继承自 BaseFinetuning，利用了 PL 的微调回调机制。
    finetune_function 留空是因为我们选择从头到尾保持骨干冻结，
    不需要在某个 epoch 解冻。
    """

    def __init__(self, unfreeze_backbone_at_epoch: int = 5, train_bn: bool = True, backbone_lr: float = 1e-5):
        super(MyBackboneFinetuning, self).__init__()

    def freeze_clip(self, modules):
        """
        递归展开所有子模块，逐一冻结。
        （方法名里的 clip 是历史遗留，实际冻结的是整个 ViLT 模型）
        """
        _modules = BaseFinetuning.flatten_modules(modules)
        for mod in _modules:
            self.freeze_module(mod)

    def freeze_module(self, module):
        """
        冻结单个模块的全部参数：将 requires_grad 设为 False。
        被注释掉的 LayerNorm 处理表明曾尝试对 LN 做特殊处理（关闭 running stats），
        但最终选择直接冻结。
        """
        # if isinstance(module, LayerNorm):
        #     module.track_running_stats = False
        for param in module.parameters(recurse=False):
            param.requires_grad = False

    def freeze_before_training(self, pl_module):
        """
        【关键】训练开始前触发：冻结 pl_module.model（即 ViLTransformerSS 的编码器部分）。
        这确保了 backbone 权重在整个训练过程中保持不变，
        只有任务头（如分类头、检索头）会更新。
        """
        self.freeze_clip(pl_module.model)

    def finetune_function(
            self, pl_module: "pl.LightningModule", epoch: int, optimizer: Optimizer, opt_idx: int
    ) -> None:
        """
        【有意留空】每个 epoch 开始时被调用。
        不做任何解冻操作，意味着 backbone 在整个训练过程中始终保持冻结。
        如果想在第 N 个 epoch 后解冻，在这里写解冻逻辑即可。
        """
        pass


@ex.automain
def main(_config):
    """
    主入口。sacred 的 @ex.automain 装饰器会自动解析命令行参数并注入 _config。
    """
    # ---- 配置深拷贝，防止对 sacred 配置对象的原地修改造成副作用 ----
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    # ---- 数据模块：负责加载多任务数据集（VQA, NLVR2, COCO 等） ----
    dm = MTDataModule(_config, dist=False)

    # ---- 模型：ViLT 的 Vision+Language Transformer ----
    model = ViLTransformerSS(_config)
    exp_name = f'{_config["exp_name"]}'

    # ---- 日志目录 & 回调设置 ----
    os.makedirs(_config["log_dir"], exist_ok=True)

    # 模型检查点：按 val/the_metric 最大值保存最佳模型
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )

    # TensorBoard 日志：路径中包含 seed 和加载权重的文件名，便于区分不同实验
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
    )

    # 学习率监控：每步记录一次
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")

    # 【关键】回调列表：MyBackboneFinetuning 排在首位，确保训练前先冻结 backbone
    callbacks = [MyBackboneFinetuning(), checkpoint_callback, lr_callback]
    # callbacks = [checkpoint_callback, lr_callback]  # 不冻结版本（被注释掉，方便切换）

    # ---- GPU 数量解析：配置中可能是 int 或 list（指定 GPU ID） ----
    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    # ---- 梯度累积步数 = 总 batch_size / (单卡 batch_size × GPU 数 × 节点数) ----
    # 这样在显存不足以支撑大 batch 时，通过累积梯度模拟等效的大 batch 训练
    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )
    print(grad_steps)

    # 如果配置了 max_steps，则按步数训练（epoch 设为 1000 兜底）；否则按 max_epoch 训练
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None
    print("this is valvalvla:{}".format(_config["val_check_interval"]))

    # ---- Trainer 配置 ----
    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        # devices=1,
        # strategy='ddp',
        num_nodes=_config["num_nodes"],
        # precision=_config["precision"],
        accelerator="ddp",            # 分布式数据并行
        # accelerator="gpu",
        # benchmark=True,
        # deterministic=True,
        max_epochs=_config["max_epoch"] if max_steps is None else 1000,
        # max_steps=max_steps,         # 按步数训练时取消注释即可
        callbacks=callbacks,
        logger=logger,
        # prepare_data_per_node=False,
        # replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,  # 梯度累积实现大等效 batch
        # log_every_n_steps=10,
        # flush_logs_every_n_steps=10,
        num_sanity_val_steps=0,              # 不执行 sanity check，直接开始训练
        resume_from_checkpoint=_config["resume_from"],
        weights_summary="top",
        # fast_dev_run=_config["fast_dev_run"],
        # auto_lr_find=True,
        # auto_scale_batch_size="power"
        # val_check_interval=1,
        val_check_interval=_config["val_check_interval"],
    )

    # ---- 训练 / 测试流程 ----
    if not _config["test_only"]:
        # 1. 训练（backbone 被 MyBackboneFinetuning 冻结，仅头部更新）
        trainer.fit(model, datamodule=dm)
        # 2. 测试：用 last checkpoint
        trainer.test(model, test_dataloaders=dm)
        # 3. 测试：用 val/the_metric 最佳的 checkpoint
        trainer.test(ckpt_path="best", test_dataloaders=dm)
    else:
        # 仅测试模式：跳过训练，直接加载权重做推理
        trainer.test(model, datamodule=dm)
