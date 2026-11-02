import numpy as np

## 矩阵乘法
a = np.array([
    [2,3,4],
    [5,6,7]
    ])

b = np.array([
    [1,2,3],
    [4,5,6]
    ])

x = a * b


c = np.array([
    [1,2],
    [3,4],
    [5,6]
    ])

d = np.array([
    [1,2,3],
    [4,5,6]
    ])

dot = c @ d
print(dot)