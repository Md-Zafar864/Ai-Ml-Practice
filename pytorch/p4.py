import torch
a=torch.randn(2,4)
b=torch.randn(4,5)
print(a)
print(b)
j=a@b 
print(j)

x=torch.randn(1,4,5)
print(x)