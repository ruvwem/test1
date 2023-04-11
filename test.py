import numpy as np
#import random
import os
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# 设置随机数种子
def seed_torch(seed=100):
	# random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

# 读入数据，并转化为numpy
def load_data(data_path = 'data/', n_components=100):
    # 遍历文件夹中的所有文件
    file_list = os.listdir(os.path.join(data_path, 'data_science'))

    verbs = pickle.load(open(os.path.join(data_path, 'verbs.pkl'), 'rb'))
    word2features = pickle.load(open(os.path.join(data_path, 'word2features.pkl'), 'rb'))
    # 按照数字顺序对文件进行排序
    file_list.sort(key=lambda x: str(x.split(".")[0]))

    data_science_data = []
    data_science_label = []
    # 读取所有文件中的数据
    for file_name in file_list:
        file_path = os.path.join(data_path, 'data_science', file_name)
        f_read = open(file_path, 'rb')
        data = pickle.load(f_read)
        label_name = data.keys()

        fMRI_data = []
        fMRI_label = []
        for k, v in data.items():
            y = word2features[k]
            for x in v:
                fMRI_data.append(np.squeeze(x))
                fMRI_label.append(y)#为什么要写重复的？
        # 使用PCA降维
        pca = PCA(n_components=n_components)
        fMRI_data = np.array(fMRI_data)
        fMRI_data = pca.fit_transform(fMRI_data)
        data_science_data.append(fMRI_data)
        data_science_label.append(np.array(fMRI_label))
    return verbs, word2features, data_science_data, data_science_label, label_name

# 划分训练、测试数据
def leave_out(data_data, data_label, leave):
    train_data, train_label = [], []
    test_data, test_label = [], []
    for i in range(60):
        data = data_data[i*6:(i+1)*6,:]
        label = data_label[i*6:(i+1)*6,:]
        if i in leave:
            test_data.append(data)
            test_label.append(label[0])
        else:
            train_data.append(data)
            train_label.append(label)
    #train_data = torch.tensor(np.concatenate(train_data), dtype=torch.float32).cuda()
    #train_label = torch.tensor(np.concatenate(train_label), dtype=torch.float32).cuda()
   # test_data = torch.tensor(np.concatenate(test_data), dtype=torch.float32).cuda()
    train_data = torch.tensor(np.concatenate(train_data), dtype=torch.float32).cpu()
    train_label = torch.tensor(np.concatenate(train_label), dtype=torch.float32).cpu()
    test_data = torch.tensor(np.concatenate(test_data), dtype=torch.float32).cpu()
    test_label = np.array(test_label)

    return train_data, train_label, test_data, test_label

# 映射网络
class MLP(nn.Module):
    def __init__(self, in_dim=360, emb_dim=720, out_dim=25):
        super(MLP, self).__init__()

        self.linear1 = nn.Sequential(
                    nn.Linear(in_dim, emb_dim), 
                    nn.BatchNorm1d(emb_dim),
                    nn.ReLU())

        self.linear2 = nn.Sequential(
                    nn.Linear(emb_dim, emb_dim), 
                    nn.BatchNorm1d(emb_dim),
                    nn.ReLU())

        self.linear4 = nn.Sequential(
                    nn.Linear(emb_dim, out_dim), 
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU())

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear4(x)
        return x

# 训练网络
def train(train_data, train_label, batch_size=64, learning_rate=0.01, num_epochs=10, f=None):
    # 初始化 MLP 模型
    #model = MLP().cuda()
    model = MLP().cpu()

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 创建 DataLoader
    dataset = TensorDataset(train_data, train_label)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    # 训练 MLP 模型
    for epoch in range(num_epochs):
        train_loss=0.0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()  # 清空梯度
            y_pred = model(X_batch)  # 前向传播
            loss = criterion(y_pred, y_batch)  # 计算损失
            train_loss+=loss.item()
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

    return model

# 测试网络
def test(model, knn, test_data, test_label, class_label, f, path, i):
    with torch.no_grad():
        outputs = model(test_data)
        outputs = outputs.cpu().numpy()

        # 保存预测向量与实际向量
        np.save(os.path.join(path, str(i)+'-outputs.np') ,outputs)
        np.save(os.path.join(path, str(i)+'-test_label.np') ,test_label)
        # print(outputs, test_label)

        # 预测分类
        predictions = knn.predict(outputs)
        accuracy = (predictions==class_label).mean()
        # 输出结果
        print(predictions, class_label)
        f.writelines([str(predictions), str(class_label),'\n'])
        return accuracy

def leave_two_out():
    # 选择预测index (默认leave-0 标签为0; leave-1 标签为1)
    # leave = random.sample(range(0, 60), 2) # 实际使用
    leave = [5, 16] # 复现结果
    if leave[1] < leave[0]:
        leave[0], leave[1] = leave[1], leave[0]
    class_label_knn = np.array([0, 1])
    class_label_test = np.array([0]*6 + [1]*6)

    # 保存结果
    path = f'leave_two_out_{leave[0]}_{leave[1]}'
    os.mkdir(path)
    f = open(os.path.join(os.path.join(path, 'log.txt')), 'w') 
    print(leave)
    f.writelines([str(leave),'\n'])

    accs = []

    # 读入数据
    verbs, word2features, data_science_data, data_science_label, label_name = load_data(n_components=360)

    for i in range(9):
        print('--------------------------')
        f.writelines(['-------------------------','\n'])
        # 划分数据
        train_data, train_label, test_data, test_label = leave_out(data_science_data[i], data_science_label[i], leave = leave)
        # 训练
        model = train(train_data, train_label, batch_size=8, learning_rate=1e-5, num_epochs=20, f=f)
        # 测试
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(test_label, class_label_knn)
        acc = test(model, knn, test_data, test_label, class_label_test, f, path, i)
        accs.append(acc)
        # 保存结果
        print(str(acc))
        f.writelines([str(acc),'\n'])

    mean = np.mean(accs)
    print('mean:', mean)
    f.writelines([str(mean),'\n'])
    return mean


def leave_one_out():
    # 选择预测index
    # leave = random.sample(range(0, 60), 1) # 实际使用
    leave = [31] # 复现结果
    class_label_knn = np.arange(0, 60)
    class_label_test =  np.array(leave*6)

    # 保存结果
    path = f'leave_one_out_{leave[0]}'
    os.mkdir(path)
    f = open(os.path.join(os.path.join(path, 'log.txt')), 'w') 
    print(leave)
    f.writelines([str(leave),'\n'])

    accs = []

    # 读入数据
    verbs, word2features, data_science_data, data_science_label, label_name = load_data(n_components=360)

    for i in range(9):
        print('--------------------------')
        f.writelines(['--------------------------','\n'])
        # 划分数据
        train_data, train_label, test_data, test_label = leave_out(data_science_data[i], data_science_label[i], leave = leave)
        # 训练
        model = train(train_data, train_label, batch_size=8, learning_rate=1e-5, num_epochs=20, f=f)
        # 测试
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(data_science_label[i][::6,:], class_label_knn)
        acc = test(model, knn, test_data, test_label, class_label_test, f, path, i)
        accs.append(acc)
        # 保存结果
        print(str(acc))
        f.writelines([str(acc),'\n'])

    mean = np.mean(accs)
    print('mean:', mean)
    f.writelines([str(mean),'\n'])
    return mean


if __name__ == '__main__':
    seed_torch()
    leave_two_out()
    leave_one_out()
