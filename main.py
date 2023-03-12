from data import load_data, preprocess_data
from model import CandlestickNet
from train import train
from test import test


#我们从 data.py、model.py、train.py 和 test.py 文件中导入必要的函数，
# 并调用这些函数来加载和预处理数据、定义模型以及训练和测试模型。

#然后我们定义模型的超参数，包括输入大小、隐藏大小、输出大小、学习率和轮数。
# 我们创建 CandlestickNet 模型，定义二元交叉熵损失函数，并初始化 Adam 优化器。
# 我们使用训练数据加载器训练模型并使用测试数据加载器测试模型。最后，我们打印出模型预测的准确性。

#请注意，这只是一个一般示例，可能需要进行自定义以满足您的特定需求和项目的性质。
# 仔细评估模型在测试数据上的性能非常重要，以确保它可以很好地泛化到新的、看不见的数据。


# Load and preprocess the data
data = load_data('candlestick_data.csv')
train_loader, test_loader = preprocess_data(data)

# Define the hyperparameters
input_size = 5
hidden_size = 10
output_size = 1
learning_rate = 0.001
num_epochs = 10

# Define the model, loss function, and optimizer
model = CandlestickNet(input_size, hidden_size, output_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train(model, train_loader, criterion, optimizer, num_epochs)

# Test the model
accuracy = test(model, test_loader)
print('Accuracy: {:.2f}%'.format(accuracy))
