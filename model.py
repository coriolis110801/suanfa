import torch
import torch.nn as nn


#我们定义了一个名为 CandlestickNet 的类，它继承自 PyTorch 中的 nn.Module 类。
# __init__ 方法定义了神经网络的层，包括输入层、一个具有 ReLU 激活函数的隐藏层和一个具有 sigmoid 激活函数的输出层。
# 前向方法定义了神经网络的前向传递，它将线性变换和激活函数应用于输入数据。

#请注意，这只是一个一般示例，可能需要进行自定义以满足您的特定需求和项目的性质。
# 仔细设计您的神经网络架构以确保它可以有效地从输入数据中学习并生成准确的预测非常重要。
# Define the neural network architecture
class CandlestickNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CandlestickNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
