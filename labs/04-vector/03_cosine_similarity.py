import torch

a = torch.tensor([3.0, 4.0])
b = torch.tensor([4.0, 3.0])

dot = torch.dot(a, b)

norm_a = torch.linalg.vector_norm(a)
norm_b = torch.linalg.vector_norm(b)

similarity = dot / (norm_a * norm_b)

print("a:", a)
print("b:", b)
print("dot:", dot.item())
print("norm_a:", norm_a.item())
print("norm_b:", norm_b.item())
print("cosine similarity:", similarity.item())
