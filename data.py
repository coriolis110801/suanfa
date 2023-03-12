import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader





#我们定义了两个函数：load_data 和 preprocess_data。
# load_data 函数从 CSV 文件或数据库加载数据并返回 Pandas DataFrame。
# preprocess_data 函数接收原始数据并对其进行预处理，以便在 PyTorch 中使用。

#preprocess_data 函数删除任何缺少数据的行，为目标变量（向上或向下）创建一个新列，为输入特征选择相关列，
# 并将数据转换为 PyTorch 张量。它还将数据拆分为训练集和测试集，并为每个集创建 PyTorch 数据集和数据加载器。
# Load the data from a CSV file or database
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Preprocess the data for use in PyTorch
def preprocess_data(data):
    # Drop any rows with missing data
    data = data.dropna()

    # Create a new column for the target variable (up or down)
    data['target'] = (data['close'] - data['open']) > 0

    # Select the relevant columns for the input features
    features = ['open', 'close', 'high', 'low', 'volume']
    X = data[features].values

    # Convert the target variable to a binary label
    y = data['target'].astype(int).values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert the data to PyTorch tensors
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()

    # Create PyTorch datasets and data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader
