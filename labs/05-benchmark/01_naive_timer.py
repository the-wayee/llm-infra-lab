import time

def task():
    # 模拟计算
    total = 0
    for i in range(1000000):
        total += i ** 2 
    return total

start = time.perf_counter()
res = task()

end = time.perf_counter()
elaped = end - start 

print("Result:", res)
print("Elaped time:", elaped, " seconds")

