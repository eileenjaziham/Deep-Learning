import torch

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

a = torch.Tensor([1.0])
b = torch.Tensor([1.0])
c = torch.Tensor([0])
a.requires_grad = True
b.requires_grad = True
c.requires_grad = True

def forward(x):
    return a*x**2+b*x+c

def loss(x,y):
    y_pre = forward(x)
    return (y_pre - y)**2

print("predict(before training)", 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print('\tgrad:', x, y, a.grad.item(),b.grad.item(),c.grad.item())
        a.data = a.data - 0.01 * a.grad.data
        b.data = b.data - 0.01 * b.grad.data
        c.data = c.data - 0.01 * c.grad.data
        a.grad.data.zero_()
        b.grad.data.zero_()
        c.grad.data.zero_()

    print("progress:", epoch, l.item())
print("predict(after training)", 4, forward(4).item())
        
