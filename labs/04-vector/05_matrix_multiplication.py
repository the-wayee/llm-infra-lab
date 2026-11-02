import torch


def main():
    a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=torch.float64)
    b = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=torch.float64)
    # a: [2,3], b: [3,2], c: [2,2]
    c = a @ b
    manual = torch.zeros(2, 2, dtype=a.dtype)
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            for k in range(a.shape[1]):
                manual[i, j] += a[i, k] * b[k, j]
    expected = torch.tensor([[22., 28.], [49., 64.]], dtype=a.dtype)
    torch.testing.assert_close(c, manual)
    torch.testing.assert_close(c, expected)
    torch.testing.assert_close(torch.mm(a, b), c)
    print("A@B=", c)
    # 内层维度不匹配是错误，不会自动转置。
    try:
        _ = a @ a
    except RuntimeError:
        print("expected: [2,3] @ [2,3] is invalid")
    else:
        raise AssertionError("invalid matrix product should fail")

    # Linear 层通常存 weight: [out_features, in_features]。
    # 输入 [batch,in]，因此 x @ weight.T + bias 得到 [batch,out]。
    #权重
    weight = torch.tensor([[1., 0., -1.], [0., 1., 1.]], dtype=a.dtype)
    # 偏置
    bias = torch.tensor([0.5, -0.5], dtype=a.dtype)
    linear = torch.nn.Linear(3, 2, dtype=a.dtype)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    actual = linear(a)
    reference = a @ weight.transpose(0, 1) + bias
    torch.testing.assert_close(actual, reference)
    print("Linear(A)=", actual.detach())

    # 批量矩阵乘法：[2,2,3] @ [2,3,2] -> [2,2,2]
    batch_a = torch.stack([a, a * 2])
    batch_b = torch.stack([b, b])
    batch_c = torch.bmm(batch_a, batch_b)
    assert tuple(batch_c.shape) == (2, 2, 2)
    torch.testing.assert_close(batch_c[0], c)
    torch.testing.assert_close(batch_c[1], c * 2)

    # 单个 b 在 batch 维广播：matmul 可以做，bmm 要求双方都是 3D。
    broadcast_c = torch.matmul(batch_a, b)
    torch.testing.assert_close(broadcast_c, batch_c)
    print("batched shape=", tuple(batch_c.shape))
    print("lesson04 experiment02 PASS")


if __name__ == "__main__":
    main()
