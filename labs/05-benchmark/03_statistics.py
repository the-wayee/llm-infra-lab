import numpy as np 

times = np.array([
    9, 10, 10, 11, 12,
    13, 14, 15, 16, 60
], dtype=np.float64)

print("===原始数据===")
print(times)

print("\n===统计数据===")
print(f"平均值:{np.mean(times):.3f} ms")
print(f"中位数:{np.median(times):.3f} ms")
print(f"最小值:{np.min(times):.3f} ms")
print(f"最大值:{np.max(times):.3f} ms")
print(f"标准差:{np.std(times):.3f} ms")

print(f"P50: {np.percentile(times, 50):.3f} ms")
print(f"P90: {np.percentile(times, 90):.3f} ms")
print(f"P99: {np.percentile(times, 99):.3f} ms")