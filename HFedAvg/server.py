import csv
import os
import argparse

from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from torch import optim

from HFedAvg.signals import get_e
from Models import Mnist_2NN, Mnist_CNN, breast_net, heart_net, phone_net
from clients import ClientsGroup, client
from model.WideResNet import WideResNet
from getData import GetDataSet
from kmeans import k_means

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
parser.add_argument('-ncomm', '--num_comm', type=int, default=200, help='number of communications')
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


def get_b_n():
    return np.random.uniform(1, 2, 20)


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
    myClients = ClientsGroup(args['dataset'], args['IID'], args['num_of_clients'], dev)
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
    all_clients =[]
    for client, label in zip(myClients.clients_set.keys(), cluster_labels):
        all_clients.append(client)

    result = []
    # num_comm 表示通信次数，此处设置为1k
    # 通讯次数一共1000次
    acc = []
    new_global_parameters =None
    b_n=get_b_n()
    mse_parameters=None
    for i in range(args['num_comm']):
        print("communicate round {}".format(i + 1))
        mse_parameters = None
        cluster_global_models = {}
        sum_parameters = None  # 用于累积所有客户端的模型参数
        if new_global_parameters is not None:
            global_parameters= new_global_parameters
        new_global_parameters = None
        for cluster_label, clients in enumerate(all_clients):
            # 客户端本地更新模型参数
            local_parameters = myClients.clients_set[client].localUpdate(
                args['epoch'],
                args['batchsize'],
                net,
                loss_func,
                opti,
                global_parameters
            )
            if mse_parameters is None:
                mse_parameters = local_parameters.copy()
            else:
                mse_parameters = {key: mse_parameters[key] + local_parameters[key] for key in mse_parameters if
                                  key in local_parameters}

            local_parameters = {key: val + get_e() for key, val in local_parameters.items()}

            # 将本地更新的模型参数累加到 sum_parameters
            if sum_parameters is None:
                sum_parameters = {key: val.clone()*b_n[cluster_label] for key, val in local_parameters.items()}
            else:
                for key in local_parameters:
                    sum_parameters[key] += local_parameters[key]*b_n[cluster_label]
        # 全局联邦学习
        # 聚合所有节点的全局模型参数
        new_global_parameters = {key: val.clone()/np.sum(b_n) for key, val in sum_parameters.items()}
        mse=0
        for key in new_global_parameters:
            mse_parameters[key] /= args['num_of_clients']
            mse += torch.sum((mse_parameters[key] - new_global_parameters[key]) ** 2).item()
        '''
            训练结束之后，我们要通过测试集来验证方法的泛化性，
            注意:虽然训练时，Server没有得到过任何一条数据，但是联邦学习最终的目的
            还是要在Server端学习到一个鲁棒的模型，所以在做测试的时候，是在Server端进行的
        '''
        # with torch.no_grad():
        # 通讯的频率
        # if (i + 1) % args['val_freq'] == 0:
        #  加载Server在最后得到的模型参数
        net.load_state_dict(new_global_parameters, strict=True)
        print(mse)
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

    df = pd.DataFrame([acc], columns=[f'Column_{i + 1}' for i in range(len(acc))])
    episodes_list = list(range(len(result)))
    plt.plot(episodes_list, result)
    plt.xlabel('communicate round')
    plt.ylabel('accuracy')
    plt.title('无优化')
    plt.show()
    csv_file_path = "one_step.csv"
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # 将数组中的每个元素作为一行写入CSV文件的一列
        for value in result:
            writer.writerow([value])
        csvfile.flush()