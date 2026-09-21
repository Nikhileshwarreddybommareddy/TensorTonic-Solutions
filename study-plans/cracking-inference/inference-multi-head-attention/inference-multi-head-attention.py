import torch
import torch.nn.functional as F

def multi_head_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as hidden_states.
    """
    X = hidden_states
    d_model = X.shape[-1]
    
    Q = X @ w_q
    K = X @ w_k
    V = X @ w_v

    d_head = d_model//num_heads
    # reformatting information for heads and restructuring information to 
    # (B,S,H,D) -> (B,H,S,D)
    Q_heads = Q.reshape(Q.shape[0], Q.shape[1], num_heads, d_head).transpose(1, 2)
    K_heads = K.reshape(K.shape[0], K.shape[1], num_heads, d_head).transpose(1, 2)
    V_heads = V.reshape(V.shape[0], V.shape[1], num_heads, d_head).transpose(1, 2)

    K_T = K_heads.transpose(-2,-1)
    Q_K_T = (Q_heads) @ (K_T)
    d_k = K_heads.shape[-1]
    scaled_Q_K_T = (Q_K_T) / (d_k**0.5)

    if causal:
        S = Q_heads.shape[-2]
        mask = torch.triu(
                torch.ones(S, S, dtype=torch.bool),
                diagonal=1,
            )
        scaled_Q_K_T = scaled_Q_K_T.masked_fill(mask, float("-inf"))

    h_i = F.softmax(scaled_Q_K_T,dim=-1) @ V_heads
    h_i = h_i.transpose(1, 2)
    h_concat = h_i.reshape(
        h_i.shape[0],
        h_i.shape[1],
        d_model
    )

    output = h_concat @ w_o

    return output

    
    

    
    

    
