import torch
import torch.nn.functional as F

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask = None,
) -> torch.Tensor:
    """
    Returns a float32 attention tensor with shape (batch, query length, value width).
    """
    k_t = key.transpose(-2,-1)
    q_k_t = query @ k_t
    d_k = key.shape[-1]
    scaled_score = ((q_k_t/d_k**0.5))
    if mask is None:
        scores = scaled_score
    else:
        scores = scaled_score.masked_fill(mask, float("-inf"))
    soft_max = F.softmax(scores,dim=-1)
    y = soft_max @ value
    return y
