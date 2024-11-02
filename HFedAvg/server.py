import copy
import os
import argparse
from random import random

from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from torch import optim
from Models import Mnist_2NN, Mnist_CNN, breast_net, heart_net, phone_net
from clients import ClientsGroup, client
from model.WideResNet import WideResNet
from getData import GetDataSet
from kmeans import k_means
from signals import single_channel, get_e

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
# 客户端的数量
parser.add_argument('-cn', '--num_of_clusters', type=int, default='4', help='num of the clusters')
parser.add_argument('-nc', '--num_of_clients', type=int, default=20, help='numer of the clients')
# 随机挑选的客户端的数量
parser.add_argument('-cf', '--cfraction', type=float, default=1,
                    help='C fraction, 0 means 1 client, 1 means total clients')
# 训练次数(客户端更新次数)
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
# batchsize大小
parser.add_argument('-B', '--batchsize', type=int, default=150, help='local train batch size')
# 模型名称
parser.add_argument('-mn', '--model_name', type=str, default='phone_net', help='the model to train')
# 学习率
parser.add_argument('-lr', "--learning_rate", type=float, default=0.001, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-dataset', "--dataset", type=str, default="phone", help="需要训练的数据集")
# 模型验证频率（通信频率）
parser.add_argument('-vf', "--val_freq", type=int, default=50, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=100, help='global model save frequency(of communication)')
# n um_comm 表示通信次数，此处设置为1k
parser.add_argument('-ncomm', '--num_comm', type=int, default=100, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=1, help='the way to allocate data to clients')


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def getbound(anchor):
    l = len(anchor)
    a = []
    lb = np.empty(l)
    ub = np.empty(l)
    for i in range(l):
        # anchor[i] = anchor[i].replace(' ','')
        anchor[i] = anchor[i].replace('>=', '>')
        anchor[i] = anchor[i].replace('<=', '<')
        anchor[i] = anchor[i].replace(' < ', '<')
        anchor[i] = anchor[i].replace(' > ', '>')
        t = anchor[i].count('<')
        if t == 0:
            anchor[i] = anchor[i].split('>')
            a.append(anchor[i][0])
            lb[i] = anchor[i][1]
            ub[i] = 1
        elif t == 1:
            anchor[i] = anchor[i].split('<')
            a.append(anchor[i][0])
            ub[i] = anchor[i][1]
            lb[i] = 0
        elif t == 2:
            anchor[i] = anchor[i].split('<')
            a.append(anchor[i][1])
            lb[i] = anchor[i][0]
            ub[i] = anchor[i][2]
    return a, lb, ub


def cal_eta(dis,b_n,dis_r_bs,b_r):
    h = [x ** (-2) * 0.001 for x in dis ]
    h_r_bs = [x ** (-2) * 0.001 for x in dis_r_bs ]
    sum_hn_bn = 0
    sum_hn2_bn2 = 0

    for i in range(len(h)):
        sum_hn2_bn2+=h[i]**2*b_n[i]**2
        sum_hn_bn+=h[i]*b_n[i]
    sum_hr2_br2=0
    for i in range(len(h_r_bs)):
        sum_hr2_br2+=h_r_bs[i]**2*b_r[i]**2
    noise = 10e-5
    return (sum_hn2_bn2+noise+noise/sum_hr2_br2)/sum_hn_bn

if __name__ == "__main__":
    args = parser.parse_args()
    args = args.__dict__

    # -----------------------文件保存-----------------------#
    # 创建结果文件夹
    # test_mkdir("./result")
    # path = os.getcwd()
    # 结果存放test_accuracy中
    test_txt = open("test_accuracy.txt", mode="a")
    # global_parameters_txt = open("global_parameters.txt",mode="a",encoding="utf-8")
    # ----------------------------------------------------#
    # 创建最后的结果
    test_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net = None
    # 初始化模型
    # mnist_2nn
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN()
    # mnist_cnn
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()
    # ResNet网络
    elif args['model_name'] == 'wideResNet':
        net = WideResNet(depth=28, num_classes=10).to(dev)
    elif args['model_name'] == 'breast_net':
        net = breast_net()
    elif args['model_name'] == 'heart_net':
        net = heart_net()
    elif args['model_name'] == 'phone_net':
        net = phone_net()

    ## 如果有多个GPU
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)

    # 将Tenor 张量 放在 GPU上
    net = net.to(dev)

    '''
        回头直接放在模型内部
    '''
    # 定义损失函数
    loss_func = F.cross_entropy
    # 优化算法的，随机梯度下降法
    # 使用Adam下降法
    opti = optim.Adam(net.parameters(), lr=args['learning_rate'])

    ## 创建Clients群
    '''
        创建Clients群100个
        
        得到Mnist数据
        
        一共有60000个样本
        100个客户端
        IID：
            我们首先将数据集打乱，然后为每个Client分配600个样本。
        Non-IID：
            我们首先根据数据标签将数据集排序(即MNIST中的数字大小)，
            然后将其划分为200组大小为300的数据切片，然后分给每个Client两个切片。
            注： 我觉得着并不是真正意义上的Non—IID
    '''
    # 初始化客户端组对象
    # ClientsGroup是一个用于管理客户端数据的类。通过传入数据集、IID、客户端数量和设备信息来创建一个客户端组实例。
    # 这里主要为了说明如何实例化这个类，以及实例化后如何获取测试数据加载器。
    myClients = ClientsGroup(args['dataset'], args['IID'], args['num_of_clients'], dev)

    # 获取客户端组的测试数据加载器
    # 通过myClients对象的test_data_loader属性，可以获取用于测试的数据加载器。
    # 这对于后续的模型验证和测试是非常重要的。
    testDataLoader = myClients.test_data_loader

    # ---------------------------------------以上准备工作已经完成------------------------------------------#
    # 每次随机选取10个Clients
    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    # 得到全局的参数
    global_parameters = {}
    # net.state_dict()  # 获取模型参数以共享

    # 得到每一层中全连接层中的名称fc1.weight
    # 以及权重weights(tenor)
    # 得到网络每一层上
    for key, var in net.state_dict().items():
        # print("key:"+str(key)+",var:"+str(var))
        print("张量的维度:" + str(var.shape))
        print("张量的Size" + str(var.size()))
        global_parameters[key] = var.clone()

    # Metric为聚类的依据，可为可解释性矩阵，距离等，此处自定义了聚类指标
    np.random.seed(0)  # 设置随机种子以确保结果的可重复性

    # 生成随机特征数据
    feature_0 = np.random.randint(low=100, high=1000, size=args['num_of_clients'])  # 数据点数量
    feature_1 = np.random.uniform(low=0.1, high=0.9, size=args['num_of_clients'])  # 正类样本比例
    feature_2 = np.random.uniform(low=10, high=100, size=args['num_of_clients'])  # 网络带宽
    feature_3 = np.random.uniform(low=100, high=500, size=args['num_of_clients'])  # 存储容量

    # 将特征组合成一个二维数组（每个客户端一行）
    Metric = np.array([feature_0, feature_1, feature_2, feature_3]).T

    result =[]
    '''
    
    # 使用 KMeans 对客户端进行聚类
    cluster_labels = k_means(Metric,args['num_of_clusters'])
    #cluster_labels = kmeans.fit_predict(Metric)

    # 创建簇内客户端群组
    cluster_clients = {label: [] for label in set(cluster_labels)}
    for client, label in zip(myClients.clients_set.keys(), cluster_labels):
        cluster_clients[label].append(client)
    '''
    # 使用 KMeans 对客户端进行聚类
    cluster_labels = k_means(Metric, args['num_of_clusters'])
    print(cluster_labels)

    # 创建簇内客户端群组
    # 由于 cluster_labels 已经是从 k_means 返回的整数数组，我们可以直接使用它们作为键
    cluster_clients = {label: [] for label in set(cluster_labels)}
    for client, label in zip(myClients.clients_set.keys(), cluster_labels):
        cluster_clients[label].append(client)

    #最大发射功率
    MAX_P = 100
    #接受系数
    ETA=1
    #每个节点到中继的距离
    dis = [np.random.randint(50, 200) for _ in range(len(cluster_labels))]
    dis_r_bs = [np.random.randint(100, 300) for _ in range(len(cluster_clients.items()))]
    b_n = [0]*len(cluster_labels)
    b_r = [0]*len(cluster_clients.items())
    # num_comm 表示通信次数，此处设置为1k
    # 通讯次数一共1000次
    acc = []
    parameters = [{} for _ in range(len(cluster_labels))]
    for i in range(args['num_comm']):
        print("communicate round {}".format(i + 1))

        cluster_global_models = {}  # key：簇序号，value：模型参数
        last_cluster_parameters = {}
        # 依次遍历每个簇
        for cluster_label, clients in cluster_clients.items():
            sum_parameters = None  # 用于累积簇内所有客户端的模型参数

            # 遍历簇内所有客户端
            for client in tqdm(clients, desc=f"Cluster {cluster_label}"):
                # 客户端本地更新模型参数
                local_parameters = myClients.clients_set[client].localUpdate(
                    args['epoch'],
                    args['batchsize'],
                    net,
                    loss_func,
                    opti,
                    global_parameters
                )
                #print("local:",local_parameters)
                # single_received_by_relay 中继收到的某一个节点的信号
                e = get_e()
                client_int =int(client[client.find("client") + len("client"):])
                parameters[client_int]=copy.deepcopy({key: val for key, val in local_parameters.items()})
                h_n_r=dis[client_int]**(-2)*0.001
                b_n[client_int]=min(ETA/h_n_r,MAX_P**0.5)
                single_received_by_relay = {key: single_channel(val, h_n_r, b_n[client_int], e)
                                            for key, val in local_parameters.items()}
                # 将中继收到的各个节点的模型参数累加到 sum_parameters
                if sum_parameters is None:
                    sum_parameters = single_received_by_relay
                else:
                    for key in single_received_by_relay:
                        sum_parameters[key] += single_received_by_relay[key]

            # 另一个中继对该节点的干扰
            e = get_e()
            other_relay_signal = {}
            # 如果 last_cluster_parameters 不为空，则将 last_cluster_parameters 作为其他中继的干扰信号
            if last_cluster_parameters != {}:
                other_relay_signal = {key: single_channel(val, np.random.normal(1, 0.1), 1, e)
                                      for key, val in last_cluster_parameters.items()}
            # 计算中继收到的模型参数
            e = get_e()
            # 如果其他中继的干扰信号不为空，则将 sum_parameters 加上其他中继的干扰信号和e
            if other_relay_signal != {}:
                for key in sum_parameters:
                    sum_parameters[key] += other_relay_signal[key]
            # 否则加上干扰e即可
            else:
                sum_parameters = {key: val.clone() + e for key, val in sum_parameters.items()}

            # 计算簇内全局模型参数的平均值
            if sum_parameters is not None:
                num_clients_in_cluster = len(clients)
                for key in sum_parameters:
                    sum_parameters[key] /= num_clients_in_cluster
                    cluster_global_models[cluster_label] = sum_parameters

            # last_cluster_parameters = sum_parameters

        test_sum_parameters = parameters[0]
        for i in range(args['num_of_clients']):
            if i == 0:
                continue
            for key in parameters[i].keys():
                test_sum_parameters[key] += parameters[i][key]
        # 簇间联邦学习
        # 聚合所有簇的全局模型参数
        global_parameters = None  # 这个簇的参数
        for cluster_label, cluster_params in cluster_global_models.items():
            e = get_e()
            h_r_bs = dis_r_bs[cluster_label] ** (-2) * 0.001
            b_r[cluster_label] = min(ETA / h_r_bs, MAX_P ** 0.5)

            if global_parameters is None:
                global_parameters = {key: single_channel(val.clone(), h_r_bs, b_r[cluster_label], e) for key, val in
                                     cluster_params.items()}
            else:
                for key in cluster_params:
                    global_parameters[key] += single_channel(cluster_params[key].clone(), h_r_bs, b_r[cluster_label], e)
        ETA = cal_eta(dis,b_n,dis_r_bs,b_r)
        # 计算最终全局模型的平均参数
        if global_parameters is not None:
            num_clusters = len(cluster_global_models)
            for key in global_parameters:
                global_parameters[key] /= (args['num_of_clients']*ETA)
                test_sum_parameters[key]/= args['num_of_clients']
                #print("global:",global_parameters[key])
                #print("test:  ",test_sum_parameters[key])
                #print("local: ",local_parameters[key])
                #print("cha ",test_sum_parameters[key]-local_parameters[key])
                print( torch.mean((test_sum_parameters[key] - global_parameters[key]) ** 2))
                # 如果需要添加噪声或其他操作，可以在此处进行

        # 更新中心服务器的模型参数
        net.load_state_dict(global_parameters, strict=True)

        # test_txt.write("communicate round " + str(i + 1) + str('accuracy: {}'.format(sum_accu / num)) + "\n")

        '''
            训练结束之后，我们要通过测试集来验证方法的泛化性，
            注意:虽然训练时，Server没有得到过任何一条数据，但是联邦学习最终的目的
            还是要在Server端学习到一个鲁棒的模型，所以在做测试的时候，是在Server端进行的
        '''
        # with torch.no_grad():
        # 通讯的频率
        # if (i + 1) % args['val_freq'] == 0:
        #  加载Server在最后得到的模型参数
        net.load_state_dict(global_parameters, strict=True)
        sum_accu = 0
        num = 0
        # 载入测试集
        for data, label in testDataLoader:
            data, label = data.to(dev), label.to(dev)
            preds = net(data)
            preds = torch.argmax(preds, dim=1)
            # sum_accu += (preds == label).float().mean()
            if preds == label:
                sum_accu += 1
            num += 1
        print("\n" + 'accuracy: {}'.format(sum_accu / num))
        result.append(float(sum_accu / num))

        test_txt.write("communicate round " + str(i + 1) + "  ")
        test_txt.write('accuracy: ' + str(float(sum_accu / num)) + "\n")
        # test_txt.close()

        acc.append(float(sum_accu / num))

        if (i + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['save_path'],
                                         '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                i, args['epoch'],
                                                                                                args['batchsize'],
                                                                                                args['learning_rate'],
                                                                                                args['num_of_clients'],
                                                                                                args['cfraction'])))
    episodes_list = list(range(len(result)))
    plt.plot(episodes_list, result)
    plt.xlabel('communicate round')
    plt.ylabel('accuracy')
    plt.title('无优化')
    plt.show()
    """df = pd.DataFrame([acc], columns=[f'Column_{i + 1}' for i in range(len(acc))])

    # 将 DataFrame 存储到 Excel 文件
    excel_file_path = 'output.xlsx'
    df.to_excel(excel_file_path, index=False)
    test_txt.close()"""
