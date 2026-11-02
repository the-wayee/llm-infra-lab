import time, statistics, torch

def synchronize(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()

def benchmark(device, n=1024, warmup=10, repeats=30):
    print(f"\n==== Device: {device} ====")
    # 随机种子
    torch.manual_seed(42)

    # 在指定设备上创建矩阵
    a = torch.rand(n, n , device=device)
    b = torch.rand(n, n , device=device)

    # 预热
    for _ in range(warmup):
        c = a @ b
    synchronize(device)

    times = []

    # 正式测试
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()

        c = a @ b
        end = time.perf_counter()

        elaped = end - start
        times.append(elaped * 1000)  # 转换为毫秒

    # 统计数据
    mean_time = statistics.mean(times)
    median_time = statistics.median(times)

    # 转换为近似 GFLOPS
    flops = 2 * n ** 3
    gflops = flops / (median_time * 1e-3) / 1e9
    print(f"Matrix shape: ({n}, {n})")
    print(f"Mean:   {mean_time:.3f} ms")
    print(f"Median: {median_time:.3f} ms")
    print(f"GFLOPS: {gflops:.2f}")

    # 防止后续扩展时误以为输出未使用
    print("Result shape:", tuple(c.shape))

# CPU 测试
benchmark(torch.device("cpu"))

# GPU 测试
if torch.cuda.is_available():
    benchmark(torch.device("cuda"))

elif torch.mps.is_available():
    benchmark(torch.device("mps"))

    