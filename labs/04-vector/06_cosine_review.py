import torch
import torch.nn.functional as F


def manual_cosine(a, b, eps=1e-12):
    # 对齐本课使用的 cosine API 分母保护：分别对两个范数设下限。
    norm_a = torch.linalg.vector_norm(a).clamp_min(eps)
    norm_b = torch.linalg.vector_norm(b).clamp_min(eps)
    return torch.dot(a, b) / (norm_a * norm_b)



def main():
    dtype = torch.float64
    eps = 1e-12
    a = torch.tensor([1., 2., 3.], dtype=dtype)
    b = torch.tensor([2., 4., 6.], dtype=dtype)
    expected_one = torch.tensor(1., dtype=dtype)
    manual = manual_cosine(a, b, eps)
    library = F.cosine_similarity(a, b, dim=0, eps=eps)
    torch.testing.assert_close(manual, library)
    torch.testing.assert_close(library, expected_one)
    torch.testing.assert_close(F.cosine_similarity(a, -a, dim=0, eps=eps), -expected_one)
    print("same_direction=", library.item())
    print("dot(a,b)=", torch.dot(a, b).item(), "dot(a,10*b)=", torch.dot(a, 10*b).item())

    # 人工构造向量，仅演示计算；不代表真实文本 embedding。
    labels = ["direction_x", "mostly_x", "direction_y", "opposite_x"]
    vectors = torch.tensor([[1., 0., 0.], [0.8, 0.2, 0.], [0., 1., 0.], [-1., 0., 0.]], dtype=dtype)
    query = torch.tensor([1., 0., 0.], dtype=dtype)
    # [1,3] 广播到 [4,3]，沿最后一维求相似度，返回 [4]。
    scores = F.cosine_similarity(vectors, query.unsqueeze(0), dim=-1, eps=eps)
    normalized_vectors = F.normalize(vectors, p=2, dim=-1, eps=eps)
    print("normalized_vectors=", normalized_vectors)
    print("normalized_vectors_shape", normalized_vectors.shape)
    normalized_query = F.normalize(query, p=2, dim=-1, eps=eps)
    print("normalized_query=", normalized_query)
    print("normalized_query_shape", normalized_query.shape)
    by_matmul = normalized_vectors @ normalized_query
    torch.testing.assert_close(scores, by_matmul)
    assert tuple(scores.shape) == (4,)
    values, indices = scores.topk(k=2)
    top_labels = [labels[i] for i in indices.tolist()]
    assert top_labels == ["direction_x", "mostly_x"]
    print("top2=", list(zip(top_labels, values.tolist())))

    # 全部两两组合才需要 [N,D] @ [D,N]，而非逐行配对。
    all_pairs = normalized_vectors @ normalized_vectors.transpose(0, 1)
    assert tuple(all_pairs.shape) == (4, 4)
    torch.testing.assert_close(all_pairs.diagonal(), torch.ones(4, dtype=dtype))
    torch.testing.assert_close(all_pairs, all_pairs.transpose(0, 1))

    zero = torch.zeros_like(query)
    zero_score = F.cosine_similarity(zero, query, dim=0, eps=eps)
    assert torch.isfinite(zero_score).item()
    torch.testing.assert_close(zero_score, torch.tensor(0., dtype=dtype))
    print("zero_vector_score=", zero_score.item(), "(numerical convention)")
    print("lesson04 experiment03 PASS")


if __name__ == "__main__":
    main()
