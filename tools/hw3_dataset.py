# 将数据读取
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 获取上级文件夹
parent_dir = os.path.dirname(BASE_DIR)

# print(f"上级文件夹: {parent_dir}")
# print(BASE_DIR)

#获取训练集和测试集的地址
train_path = os.path.join(parent_dir, 'timit_11', 'train_11.npy')
train_label_path = os.path.join(parent_dir, 'timit_11', 'train_label_11.npy')
test_path = os.path.join(parent_dir, 'timit_11', 'test_11.npy')

#将地址转成numpy
train_np = np.load(train_path)
train_label_np = np.load(train_label_path)
test_np = np.load(test_path)


class classification_1(Dataset):
    def __init__(self, feature_path, label_path, transform=None):
        self.feature_path = feature_path
        self.transform = transform
        self.label_path = label_path

        self.features = np.load(feature_path)
        self.labels = np.load(label_path)

        # 在初始化时统一转换所有标签
        self._convert_labels()

        #数据标准化
        self._normalize_features()

        print(f"数据集加载完成: {len(self)} 个样本")
        print(f"特征维度: {self.get_feature_dim()}")
        print(f"特征形状: {self.features.shape}")
        print(f"标签形状: {self.labels.shape}")
        print(f"标签数据类型: {self.labels.dtype}")
        print(f"唯一标签: {np.unique(self.labels)}")

    def _convert_labels(self):
        """统一转换标签为整数"""
        original_dtype = self.labels.dtype
        print(f"原始标签数据类型: {original_dtype}")

        # 如果标签是字符串类型，转换为整数
        if self.labels.dtype.kind in ['U', 'S']:  # Unicode字符串或字节字符串
            print("检测到字符串标签，正在转换为整数...")
            try:
                self.labels = self.labels.astype(np.int64)
                print("标签转换成功!")
            except ValueError as e:
                print(f"转换错误: {e}")
                # 如果直接转换失败，尝试逐个转换
                converted_labels = []
                for i, label in enumerate(self.labels):
                    try:
                        converted_labels.append(int(label))
                    except ValueError:
                        print(f"无法转换标签 '{label}' 在第 {i} 个位置")
                        converted_labels.append(0)  # 使用默认值
                self.labels = np.array(converted_labels, dtype=np.int64)

        print(f"转换后标签数据类型: {self.labels.dtype}")

    def __getitem__(self, index):
        features = self.features[index]
        label = self.labels[index]

        # 现在标签已经是数值，直接转换
        features = torch.FloatTensor(features)
        label = torch.tensor(label, dtype=torch.long)

        return features, label

    def __len__(self):
        return len(self.features)

    def get_feature_dim(self):
        return self.features.shape[1]

    def _normalize_features(self):
        """数据标准化"""
        mean = self.features.mean(axis=0)
        std = self.features.std(axis=0)
        # 避免除零
        std = np.where(std == 0, 1, std)
        return (self.features - mean) / std



if __name__ == "__main__":
    # print(train_np)
    # print(train_np.shape)
    # print(train_label_np)
    # print(train_label_np.shape)
    test = classification_1(train_path, train_label_path)
    print(test.label.shape)
