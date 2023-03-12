# Define the testing function


#我们定义了一个名为 test 的函数，它将模型和测试数据加载器作为输入。
# 测试函数运行测试循环，通过模型提供批量数据并计算模型预测的准确性。

#然后我们调用测试函数使用测试数据加载器测试训练模型并打印出模型预测的准确性。

#请注意，这只是一个一般示例，可能需要进行自定义以满足您的特定需求和项目的性质。
# 仔细评估模型在测试数据上的性能非常重要，以确保它可以很好地泛化到新的、看不见的数据。



def test(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = torch.round(outputs)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Test the model
accuracy = test(model, test_loader)
print('Accuracy: {:.2f}%'.format(accuracy))
