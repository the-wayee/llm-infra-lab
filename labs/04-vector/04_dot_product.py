import torch


def python_dot(a, b):
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return sum(x * y for x, y in zip(a, b))


def main():
    a_list = [1.0, 2.0, 3.0]
    b_list = [4.0, 5.0, 6.0]
    a = torch.tensor(a_list, dtype=torch.float64)
    b = torch.tensor(b_list, dtype=torch.float64)
    manual = python_dot(a_list, b_list)
    by_dot = torch.dot(a, b)
    by_mul_sum = (a * b).sum()
    print("manual=", manual, "dot=", by_dot.item(), "mul_sum=", by_mul_sum.item())
    assert manual == 32.0  # 1*4 + 2*5 + 3*6
    # torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(by_dot, torch.tensor(manual, dtype=a.dtype))
    torch.testing.assert_close(by_dot, by_mul_sum)
    assert by_dot.ndim == 0  # 标量 Tensor
    assert (a * b).shape == a.shape  # 逐元素乘法仍是向量

    # 点积不仅取决于方向，还受长度影响。
    torch.testing.assert_close(torch.dot(a, 2 * b), 2 * by_dot)
    u = torch.tensor([1., 0.], dtype=a.dtype)
    v = torch.tensor([0., 1.], dtype=a.dtype)
    torch.testing.assert_close(torch.dot(u, v), torch.tensor(0., dtype=a.dtype))
    print("orthogonal_dot=", torch.dot(u, v).item())
    print("opposite_dot=", torch.dot(u, -u).item())

    # 负面案例：捕获预期错误，不让整个实验中断。
    try:
        python_dot([1., 2.], [3.])
    except ValueError:
        print("expected: mismatched vector lengths")
    else:
        raise AssertionError("length check missing")
    print("lesson04 experiment01 PASS")


if __name__ == "__main__":
    main()
