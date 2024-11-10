from torch_geometric.data import HeteroData  # 导入PyTorch Geometric中的HeteroData类，用于处理异构图数据
import torch  # 导入PyTorch库
import numpy as np  # 导入NumPy库，用于数组操作
from tqdm import tqdm  # 导入tqdm库，用于显示进度条
import os.path as osp  # 导入os.path模块，用于处理文件路径
import pickle  # 导入pickle模块，用于序列化数据

def load_graph():
    print('loading graph data...')  # 打印加载图数据的提示信息

    # 初赛的相关数据文件路径
    session1_node_feat_file_path = '../data/session1/icdm2022_session1_nodes.csv'  # 初赛节点特征数据文件路径
    session1_edge_file_path = '../data/session1/icdm2022_session1_edges.csv'  # 初赛边数据文件路径
    session1_item_label_file_path = '../data/session1/icdm2022_session1_train_labels.csv'  # 初赛物品标签文件路径
    session1_test_ids_file_path = '../data/session1/icdm2022_session1_test_ids.csv'  # 初赛测试ID文件路径

    # 复赛的相关数据文件路径
    session2_node_feat_file_path = '../data/session2/icdm2022_session2_nodes.csv'  # 复赛节点特征数据文件路径
    session2_edge_file_path = '../data/session2/icdm2022_session2_edges.csv'  # 复赛边数据文件路径
    session2_test_ids_file_path = '../data/session2/icdm2022_session2_test_ids.csv'  # 复赛测试ID文件路径

    # 保存图数据的文件路径
    save_graph_path = '../data/graph.pt'  # 保存图数据的文件路径

    # 初始化映射字典
    session1_node_maps = {}  # 初赛节点映射字典，保存节点ID到图中节点顺序的映射
    session2_node_maps = {}  # 复赛节点映射字典，保存节点ID到图中节点顺序的映射

    node_feat_maps = {}  # 存储各节点类型的特征
    edge_maps = {}  # 存储边的信息，包括源节点、目标节点和边类型

    lack_num = 0  # 缺失节点特征的计数器

    # 加载初赛的节点特征数据
    print('loading session1_node_feat_file_path=', session1_node_feat_file_path)  # 打印正在加载的初赛节点特征文件路径
    with open(session1_node_feat_file_path, 'r', encoding='utf-8') as f:  # 打开初赛的节点特征文件
        for l in tqdm(f):  # 使用tqdm显示加载进度
            data = l.strip().split(",")  # 将每行数据按逗号分割
            node_id = int(data[0])  # 获取节点ID
            node_type = data[1]  # 获取节点类型

            if len(data[2]) < 50:  # 如果节点特征数据长度小于50，表示特征缺失
                node_feat = np.zeros(256, dtype=np.float32)  # 用零填充缺失的节点特征
                lack_num += 1  # 计数缺失特征的节点
            else:
                node_feat = np.array([x for x in data[2].split(":")], dtype=np.float32)  # 将特征数据转为NumPy数组

            if node_type not in session1_node_maps.keys():  # 如果节点类型没有对应的映射字典，初始化为空字典
                session1_node_maps[node_type] = {}
            if node_type not in node_feat_maps.keys():  # 如果节点类型没有对应的特征列表，初始化为空列表
                node_feat_maps[node_type] = []
            session1_node_maps[node_type][node_id] = len(session1_node_maps[node_type])  # 记录节点ID到节点顺序的映射
            node_feat_maps[node_type].append(node_feat)  # 将节点特征添加到特征列表中

    # 加载复赛的节点特征数据
    print('loading session2_node_feat_file_path=', session2_node_feat_file_path)  # 打印正在加载的复赛节点特征文件路径
    with open(session2_node_feat_file_path, 'r', encoding='utf-8') as f:  # 打开复赛的节点特征文件
        for l in tqdm(f):  # 使用tqdm显示加载进度
            data = l.strip().split(",")  # 将每行数据按逗号分割
            node_id = int(data[0])  # 获取节点ID
            node_type = data[1]  # 获取节点类型

            if len(data[2]) < 50:  # 如果节点特征数据长度小于50，表示特征缺失
                node_feat = np.zeros(256, dtype=np.float32)  # 用零填充缺失的节点特征
                lack_num += 1  # 计数缺失特征的节点
            else:
                node_feat = np.array([x for x in data[2].split(":")], dtype=np.float32)  # 将特征数据转为NumPy数组

            if node_type not in session2_node_maps.keys():  # 如果节点类型没有对应的映射字典，初始化为空字典
                session2_node_maps[node_type] = {}
            if node_type not in node_feat_maps.keys():  # 如果节点类型没有对应的特征列表，初始化为空列表
                node_feat_maps[node_type] = []
            session2_node_maps[node_type][node_id] = len(session2_node_maps[node_type]) + len(session1_node_maps[node_type])  # 复赛的节点ID继承初赛的节点ID
            node_feat_maps[node_type].append(node_feat)  # 将节点特征添加到特征列表中

    print('lack_num=', lack_num)  # 打印缺失特征的节点数量

    # 加载初赛的边信息
    print('loading session1_edge_file_path = ', session1_edge_file_path)  # 打印正在加载的初赛边数据文件路径
    with open(session1_edge_file_path, 'r', encoding='utf-8') as f:  # 打开初赛的边数据文件
        for l in tqdm(f):  # 使用tqdm显示加载进度
            data = l.strip().split(",")  # 将每行数据按逗号分割
            sour_id = int(data[0])  # 获取源节点ID
            dest_id = int(data[1])  # 获取目标节点ID
            sour_type = data[2]  # 获取源节点类型
            dest_type = data[3]  # 获取目标节点类型
            edge_type = data[4]  # 获取边的类型

            pyg_edge_type = (sour_type, edge_type, dest_type)  # 构造PyG中的边类型

            if pyg_edge_type not in edge_maps.keys():  # 如果该边类型没有对应的映射字典，初始化为空字典
                edge_maps[pyg_edge_type] = {}
                edge_maps[pyg_edge_type]['sour'] = []  # 源节点ID列表
                edge_maps[pyg_edge_type]['dest'] = []  # 目标节点ID列表

            new_sour_id = session1_node_maps[sour_type][sour_id]  # 使用初赛节点映射字典获取新源节点ID
            new_dest_id = session1_node_maps[dest_type][dest_id]  # 使用初赛节点映射字典获取新目标节点ID

            edge_maps[pyg_edge_type]['sour'].append(new_sour_id)  # 将新源节点ID添加到源节点列表中
            edge_maps[pyg_edge_type]['dest'].append(new_dest_id)  # 将新目标节点ID添加到目标节点列表中

    # 加载复赛的边信息
    print('loading session2_edge_file_path =', session2_edge_file_path)  # 打印正在加载的复赛边数据文件路径
    with open(session2_edge_file_path, 'r', encoding='utf-8') as f:  # 打开复赛的边数据文件
        for l in tqdm(f):  # 使用tqdm显示加载进度
            data = l.strip().split(",")  # 将每行数据按逗号分割
            sour_id = int(data[0])  # 获取源节点ID
            dest_id = int(data[1])  # 获取目标节点ID
            sour_type = data[2]  # 获取源节点类型
            dest_type = data[3]  # 获取目标节点类型
            edge_type = data[4]  # 获取边的类型

            pyg_edge_type = (sour_type, edge_type, dest_type)  # 构造PyG中的边类型

            if pyg_edge_type not in edge_maps.keys():  # 如果该边类型没有对应的映射字典，初始化为空字典
                edge_maps[pyg_edge_type] = {}
                edge_maps[pyg_edge_type]['sour'] = []  # 源节点ID列表
                edge_maps[pyg_edge_type]['dest'] = []  # 目标节点ID列表

            new_sour_id = session2_node_maps[sour_type][sour_id]  # 使用复赛节点映射字典获取新源节点ID
            new_dest_id = session2_node_maps[dest_type][dest_id]  # 使用复赛节点映射字典获取新目标节点ID
            edge_maps[pyg_edge_type]['sour'].append(new_sour_id)  # 将新源节点ID添加到源节点列表中
            edge_maps[pyg_edge_type]['dest'].append(new_dest_id)  # 将新目标节点ID添加到目标节点列表中

    # 创建HeteroData对象
    graph = HeteroData()  # 创建一个空的异构图对象

    # 将节点特征写入图数据
    for node_type in tqdm(node_feat_maps.keys()):  # 遍历所有节点类型
        graph[node_type].x = torch.tensor(np.array(node_feat_maps[node_type]))  # 将节点特征转换为PyTorch张量
        graph[node_type].maps = session1_node_maps[node_type]  # 添加初赛的节点ID映射
        graph[node_type].maps2 = session2_node_maps[node_type]  # 添加复赛的节点ID映射

    # 将边的信息写入图数据
    for pyg_edge_type in tqdm(edge_maps.keys()):  # 遍历所有边类型
        sour = torch.tensor(edge_maps[pyg_edge_type]['sour'], dtype=torch.long)  # 源节点ID列表
        dest = torch.tensor(edge_maps[pyg_edge_type]['dest'], dtype=torch.long)  # 目标节点ID列表
        graph[pyg_edge_type].edge_index = torch.vstack([sour, dest])  # 将源节点和目标节点合并为边的索引

    # 添加初赛的物品标签，复赛没有提供物品标签
    item_labels = np.array([-1] * graph['item'].x.shape[0])  # 初始化物品标签为-1
    with open(session1_item_label_file_path, 'r', encoding='utf-8') as f:  # 打开初赛的物品标签文件
        for l in tqdm(f):  # 使用tqdm显示加载进度
            data = l.strip().split(",")  # 将每行数据按逗号分割
            itemid = int(data[0])  # 获取物品ID
            label = int(data[1])  # 获取物品标签
            new_itemid = graph['item'].maps[itemid]  # 使用图中物品ID的映射获取新ID
            item_labels[new_itemid] = label  # 更新物品标签

    graph['item'].y = torch.tensor(item_labels, dtype=torch.long)  # 将物品标签保存到图数据中

    # 保存图数据到文件
    torch.save(graph, save_graph_path)  # 将图数据保存到指定路径
