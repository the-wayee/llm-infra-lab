import numpy as np 

# 创建原始矩阵
a = np.array([
    [1,2,3],
    [4,5,6]
])
print("a.dtype: ", a.dtype)
# 转置
b = a.T

# 创建副本
c = b.copy()

print("=== 原始矩阵 a ===")
print(a)
print("shape:", a.shape)
print("strides:", a.strides)
print("C-contiguous:", a.flags['C_CONTIGUOUS'])

print("\n=== 转置矩阵 b ===")
print(b)
print("shape:", b.shape)
print("strides:", b.strides)
print("C-contiguous:", b.flags['C_CONTIGUOUS'])

print("\n=== 副本矩阵 c ===")
print(c)
print("shape:", c.shape)
print("strides:", c.strides)
print("C-contiguous:", c.flags['C_CONTIGUOUS'])

print("\n=== 内存共享 ===")
print(" A 和 B: ", np.shares_memory(a, b))
print(" B 和 C: ", np.shares_memory(b, c))

# 修改 View
b[0, 0] = 100

print("\n修改 B 后的 A:")
print(a)

print("\n修改 B 后的 C:")
print(c)