import torch.optim as optim



#我们定义了一个名为 train 的函数，它将模型、训练数据加载器、损失函数、优化器和轮数作为输入。
# train 函数运行训练循环，它通过模型迭代地提供批量数据，计算损失，反向传播梯度，并使用优化器更新模型的权重。

#我们还定义了模型的超参数，包括输入大小、隐藏大小、输出大小、学习率和轮数。
# 然后我们定义模型、损失函数和优化器，并调用训练函数以使用训练数据加载器训练模型。

#请注意，这只是一个一般示例，可能需要进行自定义以满足您的特定需求和项目的性质。
# 仔细调整超参数并监控训练进度非常重要，以确保模型有效地从输入数据中学习。


# Define the training function
def train(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

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
