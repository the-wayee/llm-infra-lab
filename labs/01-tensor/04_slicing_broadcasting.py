import numpy as np 

# 1. 创建数组   
arr = np.arange(12).reshape(3, 4)

print("原始数组:")
print(arr)
print("Shape:", arr.shape)

# 2.切片
print("\n前两行")
print(arr[:2, :])

print("\n最后一列")
print(arr[:, -1])

# 3.广播
b = np.array([1, 2, 3, 4])
print("\n广播相加")
print(arr + b)

# 4.聚合
print("\n每列之和")
print(arr.sum(axis=0))

print("\n每行平均值")
print(arr.mean(axis=1))

# 5.元素类型
c = arr.astype(np.float32)
print("\n元素类型转换")
print(c.dtype)