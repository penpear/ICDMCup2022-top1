import os  # 导入os模块，用于与操作系统交互
import os.path as osp  # 导入os.path模块，提供文件路径操作功能
import argparse  # 导入argparse模块，用于处理命令行参数
import json  # 导入json模块，用于解析JSON数据

import numpy as np  # 导入NumPy库，用于数值计算和数组操作
import torch  # 导入PyTorch库
import torch.nn.functional as F  # 导入PyTorch的函数式API
from tqdm import tqdm  # 导入tqdm库，用于显示进度条
import torch.nn as nn  # 导入PyTorch的神经网络模块
from torch_geometric.loader import NeighborLoader  # 导入PyG的NeighborLoader类，用于加载邻居采样
from torch_geometric.nn import RGCNConv, TransformerConv  # 导入PyG的RGCNConv和TransformerConv卷积层
from sklearn.metrics import average_precision_score  # 导入sklearn的平均精度得分函数
from log_model import Logger  # 导入日志记录器
import random  # 导入random库，用于生成随机数
import pandas as pd  # 导入Pandas库，用于数据操作
import numpy as np  # 导入NumPy库
from collections import Counter  # 导入Counter类，用于计数
import pickle as pkl  # 导入pickle库，用于序列化和反序列化对象


# 预训练模型的主函数
def run_pretrain1():
    # 加载日志记录类
    logger = Logger('logs/', level='debug')  # 创建Logger对象，指定日志文件夹和日志级别

    logger.logger.info('start run pretrain1.....')  # 记录信息：开始运行预训练

    # 解析命令行参数
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument('--dataset', type=str, default='../data/graph.pt')  # 图数据集的路径
    parser.add_argument('--test_file', type=str, default='../data/session1/icdm2022_session1_test_ids.txt')  # 测试集文件路径
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size. If -1, use full graph training.")  # 批大小
    parser.add_argument("--fanout", type=int, default=300, help="Fan-out of neighbor sampling.")  # 邻居采样的输出数量
    parser.add_argument("--n-layers", type=int, default=2, help="number of propagation rounds")  # GCN层数
    parser.add_argument("--h-dim", type=int, default=768, help="number of hidden units")  # 隐藏层单元数量
    parser.add_argument("--in-dim", type=int, default=256, help="number of hidden units")  # 输入特征的维度
    parser.add_argument("--n-bases", type=int, default=8, help="number of filter weight matrices, default: -1 [use all]")  # RGCN卷积中的基数
    parser.add_argument("--n-epoch", type=int, default=500)  # 训练的轮数
    parser.add_argument("--lr", type=float, default=0.0005)  # 学习率
    parser.add_argument("--device-id", type=str, default="0")  # 选择使用的GPU设备ID

    # 解析命令行参数
    args = parser.parse_args()
    logger.logger.info(str(args))  # 记录解析后的参数信息

    # 设置设备（CUDA）
    device = 'cuda:' + str(args.device_id)
    logger.logger.info('device=' + str(device))  # 记录设备信息

    # 加载图数据集
    hgraph = torch.load(args.dataset)  # 加载图数据

    # 设置随机种子函数
    def random_seed(seed):
        """设置随机种子"""
        torch.manual_seed(seed)  # 设置PyTorch的随机种子
        torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子
        torch.cuda.manual_seed(seed)  # 设置当前GPU的随机种子
        np.random.seed(seed)  # 设置NumPy的随机种子
        random.seed(seed)  # 设置Python的随机种子

    random_seed(4096)  # 设置固定随机种子

    # 获取物品节点的标签
    node_labels = hgraph['item'].y.numpy()

    # 得到训练集和验证集的划分
    ids_0 = list(np.argwhere(node_labels == 0).reshape(1, -1)[0])  # 获取标签为0的节点索引
    ids_1 = list(np.argwhere(node_labels == 1).reshape(1, -1)[0])  # 获取标签为1的节点索引

    # 打乱节点索引
    random.shuffle(ids_0)  # 打乱标签为0的节点
    random.shuffle(ids_1)  # 打乱标签为1的节点

    # 划分训练集和验证集
    train_idx = ids_0[16161:] + ids_1[951:]  # 训练集的节点索引
    val_idx = ids_0[:16161] + ids_1[:951]  # 验证集的节点索引

    # 打乱训练集和验证集
    random.shuffle(train_idx)  # 打乱训练集索引
    random.shuffle(val_idx)  # 打乱验证集索引

    # 得到item节点的训练id
    train_idx = train_idx + val_idx  # 合并训练集和验证集
    logger.logger.info('session1 item number=' + str(len(train_idx)))  # 记录训练集节点数量

    # 加载初赛的测试集节点
    logger.logger.info('loading session1 test file=' + args.test_file)  # 记录正在加载的测试集文件路径
    test_id = [int(x) for x in open(args.test_file).readlines()]  # 读取测试集的节点ID
    converted_test_id = []
    for itemid in test_id:
        # 注意复赛的item节点用maps进行转换
        converted_test_id.append(hgraph['item'].maps[itemid])  # 使用映射将测试集节点ID转换

    test_idx = converted_test_id  # 记录转换后的测试集节点ID
    logger.logger.info('session1 item number=' + str(len(test_idx)))  # 记录测试集节点数量

    # 合并初赛训练集和测试集节点
    train_idx = train_idx + test_idx  # 训练集包含测试集节点
    random.shuffle(train_idx)  # 打乱合并后的训练集节点

    train_idx = torch.tensor(np.array(train_idx))  # 转换为PyTorch张量

    logger.logger.info('item train number=' + str(len(train_idx)))  # 记录合并后的训练集节点数量

    # 获取图中边的类型数量
    num_relations = len(hgraph.edge_types)  # 图中边的类型数量

    # 定义RGCN卷积层
    class rgcn(torch.nn.Module):
        def __init__(self, in_channels, out_channels, num_relations, n_bases):
            super().__init__()
            self.conv = RGCNConv(in_channels, out_channels, num_relations, num_bases=n_bases)  # 定义RGCN卷积层
            self.ly = nn.LayerNorm(out_channels)  # 定义层归一化

        def forward(self, x, edge_index, edge_type):
            x = self.conv(x, edge_index, edge_type)  # 应用RGCN卷积层
            x = self.ly(x)  # 应用层归一化
            return x  # 返回输出

    # 定义RGCN模型
    class RGCN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, n_layers=2):
            super().__init__()
            self.convs = torch.nn.ModuleList()  # 存储卷积层
            self.relu = F.relu  # ReLU激活函数
            self.convs.append(rgcn(in_channels, hidden_channels, num_relations, args.n_bases))  # 添加第一层RGCN卷积层
            for i in range(n_layers - 2):  # 添加中间层的RGCN卷积层
                self.convs.append(rgcn(hidden_channels, hidden_channels, num_relations, args.n_bases))
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations, args.n_bases))  # 添加最后一层RGCN卷积层

            # 特征重构层
            self.line_feat_1 = nn.Linear(hidden_channels, in_channels)  # 第一层线性变换
            self.line_feat_2 = nn.Linear(in_channels, out_channels)  # 第二层线性变换

        def forward(self, x, edge_index, edge_type):
            origin_x = x  # 保存输入特征
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index, edge_type)  # 应用RGCN卷积层
                if i < len(self.convs) - 1:  # 如果不是最后一层
                    x = x.relu_()  # 激活
                    x = F.dropout(x, p=0.4, training=self.training)  # Dropout正则化
            out_feat = self.line_feat_1(x)  # 应用特征重构层
            feat_loss_vec = (out_feat - origin_x) * (out_feat - origin_x)  # 计算特征损失
            out_feat = self.line_feat_2(out_feat)  # 第二层特征重构
            return feat_loss_vec, out_feat  # 返回损失和输出特征

    model = RGCN(in_channels=args.in_dim, hidden_channels=args.h_dim, out_channels=2, n_layers=5)  # 初始化RGCN模型
    model.to(device)  # 将模型移到指定的设备（GPU）

    # 获取模型参数
    param_optimizer = list(model.named_parameters())  # 获取模型的所有参数
    # 优化器参数分组（针对不同的权重进行不同的衰减）
    no_decay = ['bias', 'gamma', 'beta']  # 不进行衰减的参数类型
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},  # 对参数进行衰减
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}  # 对不衰减的参数不进行衰减
    ]
    # 使用AdamW优化器，设置学习率
    opt = torch.optim.AdamW(optimizer_grouped_parameters, 
                            lr=args.lr)
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=3, T_mult=2, eta_min=2e-5, last_epoch=-1)

    # 定义训练函数
    def train(labeled_class, train_loader, epoch_id, epoch, batch_size):
        show_step = 10  # 打印信息的频率
        train_loader_size = train_loader.__len__()  # 获取训练数据的大小
        num_train_step = int(len(train_loader.dataset) / batch_size)  # 计算训练步数

        model.train()  # 设置模型为训练模式
        y_pred_feat = []
        y_true = []

        total_loss = []
        for batch_id, batch in enumerate(train_loader):  # 遍历训练数据
            opt.zero_grad()  # 清空梯度
            batch_size = batch[labeled_class].batch_size  # 获取当前批次的大小

            start = 0
            for ntype in batch.node_types:
                if ntype == labeled_class:  # 如果当前是标签节点类型
                    break
                start += batch[ntype].num_nodes  # 获取节点的起始位置

            batch = batch.to_homogeneous()  # 转换为同质图（只处理一个类型的节点）
            feat_loss_vec, out_feat = model(batch.x.to(device), batch.edge_index.to(device), batch.edge_type.to(device))  # 前向传播
            feat_loss_vec = feat_loss_vec[start:start + batch_size]  # 提取特征损失向量

            loss = torch.mean(torch.sum(feat_loss_vec, dim=1))  # 计算损失
            loss.backward()  # 反向传播
            opt.step()  # 更新参数

            torch.cuda.empty_cache()  # 清空缓存
            scheduler.step(epoch_id + batch_id / train_loader_size)  # 学习率更新

            total_loss.append(loss.item())  # 记录损失
            if batch_id % show_step == 0:
                log = 'labeled_class=' + labeled_class + ' epoch_id=' + str(epoch_id) + '/' + str(epoch) + ' batch_id=' + str(batch_id) + '/' + str(num_train_step) + ' lr=' + str(opt.param_groups[-1]['lr']) + ' loss_feat=' + str(np.mean(total_loss))  # 打印训练信息
                logger.logger.info(log)  # 记录日志
                total_loss = []  # 清空损失记录

    # 创建训练数据加载器
    train_loader = NeighborLoader(hgraph, input_nodes=('item', train_idx),
                                  num_neighbors=[args.fanout] * args.n_layers,
                                  shuffle=True, batch_size=args.batch_size)
    logger.logger.info('complete train_loader.....')  # 记录训练加载器构建完成

    epoch = args.n_epoch  # 训练的轮数
    batch_size = args.batch_size  # 批次大小

    # 训练模型
    for epoch_id in range(epoch):  # 迭代训练轮次
        train('item', train_loader, epoch_id, epoch, batch_size)  # 执行训练
        # 保存模型（每2轮保存一次，避免损坏）
        model_name = osp.join('../models', "pretrain_item_" + str(epoch_id % 2) + ".pth")  # 保存模型路径
        logger.logger.info('save model = ' + model_name)  # 记录保存的模型路径
        torch.save(model.state_dict(), model_name)  # 保存模型的状态字典

    # 保存最后一次模型
    model_name = osp.join('../models', "pretrain_item.pth")  # 保存最后模型的路径
    logger.logger.info('save model = ' + model_name)  # 记录最后保存的模型路径
    torch.save(model.state_dict(), model_name)  # 保存模型
