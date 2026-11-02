import torch

a = torch.arange(12).reshape(3, 4)

b = a.permute(1, 0)

c = b.reshape(12)

print("原始tensor:")
print(a)
print("shape:", a.shape)
print("strides:", a.stride())
print("C-contiguous:", a.is_contiguous())

print("\n Permute后的tensor:")
print(b)
print("shape:", b.shape)
print("strides:", b.stride())
print("C-contiguous:", b.is_contiguous())

print("\n Reshape后的tensor:")
print(c)
print("shape:", c.shape)
print("strides:", c.stride())
print("C-contiguous:", c.is_contiguous())

print("\n修改 b[0, 0]")
b[0, 0] = 100

print("a[0, 0]:", a[0, 0].item())
print("c[0]:", c[0].item())