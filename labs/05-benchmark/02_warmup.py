import time
import numpy as np

rng = np.random.default_rng(44)

a = rng.random((512, 512), dtype=np.float32)
b = rng.random((512, 512), dtype=np.float32)

def matmul():
    return a @ b

def measure_once():
    start = time.perf_counter()
    res = matmul()
    end = time.perf_counter()

    return (end - start) * 1000

print ("==== 预热之前 ====")

for i in range(5):
    elaped = measure_once()
    print(f"第 {i+1} 次运行耗时: {elaped:.3f} ms")


print("\n=== warmup ====")
for _ in range(10):
    matmul()

print("\n=== 正式测试 ===")
results = []
for i in range(20):
    elaped = measure_once()
    results.append(elaped)
    print(f"第 {i+1} 次运行耗时: {elaped:.3f} ms")

print("\n平均耗时:")
print(f"{np.mean(results):.3f} ms")