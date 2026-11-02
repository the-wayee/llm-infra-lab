import torch


def f(x):
    return x * x + 3 * x


def main():
    x = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    y = f(x)  # forward: y=10
    y.backward()  # dy/dx = 2x+3 = 7
    print("y=", y.item(), "grad=", x.grad.item())
    torch.testing.assert_close(x.grad, torch.tensor(7.0, dtype=x.dtype))

    # 新建 forward 图后再次 backward：梯度默认累加，不会自动覆盖。
    f(x).backward()
    torch.testing.assert_close(x.grad, torch.tensor(14.0, dtype=x.dtype))
    x.grad = None
    f(x).backward()
    torch.testing.assert_close(x.grad, torch.tensor(7.0, dtype=x.dtype))

    # 中心差分：用很小的扰动近似导数，便于理解，不作为实际训练算法。
    base = 2.0
    eps = 1e-6
    finite_difference = (f(base + eps) - f(base - eps)) / (2 * eps)
    assert abs(finite_difference - 7.0) < 1e-6
    print("finite_difference=", finite_difference)

    z = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    chain = (3 * z + 1) ** 2
    grad, = torch.autograd.grad(chain, z)
    torch.testing.assert_close(grad, torch.tensor(42.0, dtype=z.dtype))
    assert z.grad is None  # autograd.grad 返回结果，不会写入本例 z.grad
    print("chain_rule_grad=", grad.item())

    with torch.no_grad():
        inference_result = f(x)
    assert not inference_result.requires_grad
    print("no_grad_result=", inference_result.item())
    print("lesson02 PASS")


if __name__ == "__main__":
    main()
