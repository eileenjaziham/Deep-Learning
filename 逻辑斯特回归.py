import numpy as np  # 导入NumPy库，用于数值计算（本代码未直接使用，但通常深度学习中会用到）
import torch  # 导入PyTorch库，核心深度学习框架
import torch.nn.functional as F  # 导入PyTorch的函数模块，包含激活函数等功能

# 定义输入数据和标签
x_data = torch.Tensor([[1.0],[2.0],[3.0]])  # 创建输入张量，形状为(3,1)，包含3个样本，每个样本1个特征
y_data = torch.Tensor([[0],[0],[1]])  # 创建标签张量，形状为(3,1)，对应二分类标签（0或1）


# 定义逻辑回归模型类
class LogisticRegressionModel(torch.nn.Module):  # 继承自torch.nn.Module（PyTorch中所有模型的基类）
    def __init__(self):  # 构造函数，初始化模型
        super(LogisticRegressionModel, self).__init__()  # 调用父类的构造函数，必须执行此步骤
        self.linear = torch.nn.Linear(1, 1)  # 创建线性层，输入维度为1（特征数），输出维度为1（预测值）
    
    def forward(self, x):  # 前向传播方法，定义模型的计算流程
        y_pred = F.sigmoid(self.linear(x))  # 先通过线性层计算线性输出，再用sigmoid函数激活（映射到0-1之间，作为概率）
        return y_pred  # 返回预测结果

model = LogisticRegressionModel()  # 创建模型实例


# 定义损失函数和优化器
criterion = torch.nn.BCELoss(size_average=False)  # 二分类交叉熵损失函数，size_average=False表示返回总损失（不平均）
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器，学习率为0.01，优化模型的所有参数


# 训练循环
for epoch in range(1000):  # 迭代训练1000轮
    y_pred = model(x_data)  # 前向传播：输入x_data，得到模型预测值y_pred（形状为(3,1)，每个元素是0-1之间的概率）
    loss = criterion(y_pred, y_data)  # 计算损失：用BCELoss比较预测值和真实标签
    print(epoch, loss.item())  # 打印当前轮次和损失值（.item()将张量转换为Python标量）
    
    optimizer.zero_grad()  # 清零梯度：避免梯度累积（PyTorch默认会累积梯度）
    loss.backward()  # 反向传播：计算损失对所有参数的梯度
    optimizer.step()  # 更新参数：根据梯度和学习率调整模型参数
