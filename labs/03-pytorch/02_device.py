import time
import torch

def benchmark(device, n = 1024, repeat = 20):
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)

    # Warm up   
    for _ in range(5):
        _ = a @ b

    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()

    start = time.perf_counter()

    for _ in range(repeat):
        c = a @ b

    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()

    elapsed = (time.perf_counter() - start) / repeat

    print(f"Device: {device}, Time per multiplication: {elapsed * 1000:.3f} ms")


benchmark(torch.device('cpu'))

if torch.cuda.is_available():
    benchmark(torch.device('cuda'))
elif torch.backends.mps.is_available():
    benchmark(torch.device('mps'))