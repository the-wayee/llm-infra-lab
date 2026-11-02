import torch


def main():
    torch.manual_seed(42)
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    print("x=", x)
    print("shape=", tuple(x.shape), "dtype=", x.dtype, "device=", x.device)
    print("elements=", x.numel(), "payload_bytes=", x.numel() * x.element_size())
    assert tuple(x.shape) == (2, 3)
    assert x.numel() == 6

    # 一维 [3] 与二维 [2,3] 的最后一维对齐，给每一行加同一个 bias。
    bias = torch.tensor([10.0, 20.0, 30.0])
    y = x + bias
    torch.testing.assert_close(y, torch.tensor([[11., 22., 33.], [14., 25., 36.]]))
    print("broadcast x+bias=", y)

    row_sum = x.sum(dim=-1)  # 每一行沿最后一维求和，[2,3] -> [2]
    torch.testing.assert_close(row_sum, torch.tensor([6., 15.]))
    print("row_sum=", row_sum)

    xt = x.transpose(0, 1)  # [2,3] -> [3,2]
    print("stride=", x.stride(), "transposed_stride=", xt.stride())
    print("contiguous=", x.is_contiguous(), xt.is_contiguous())
    flat = xt.reshape(-1)  # -1 让 PyTorch 推算这一维长度
    torch.testing.assert_close(flat, torch.tensor([1., 4., 2., 5., 3., 6.]))
    # 先显式转成连续布局，view 就能展平；它与 reshape 的数值结果一致。
    torch.testing.assert_close(xt.contiguous().view(-1), flat)

    try:
        _ = x + torch.ones(4)
    except RuntimeError:
        print("expected: [2,3] cannot broadcast with [4]")
    else:
        raise AssertionError("shape mismatch should fail")
    print("lesson01 PASS")


if __name__ == "__main__":
    main()
