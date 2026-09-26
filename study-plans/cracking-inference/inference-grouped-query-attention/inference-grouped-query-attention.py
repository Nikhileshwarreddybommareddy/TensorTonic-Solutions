import torch
import torch.nn.functional as F
def grouped_query_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns an attention tensor with the same shape as hidden_states.
    """
    x = hidden_states
    h_q = num_query_heads
    h_kv = num_kv_heads
    d_model = x.shape[-1]
    # heads values is always related to queries so no matter what we should always have 
    d_head = d_model // h_q
    if num_query_heads%num_kv_heads == 0:
        groups = num_query_heads // num_kv_heads
    else:
        raise ValueError()
        
    q = x @ w_q
    k = x @ w_k
    # dim in torch [b,s,h,d] -> [0,1,2,3]

    v = x @ w_v

    q = q.reshape(q.shape[0],q.shape[1],h_q,d_head).transpose(1,2)
    k = k.reshape(k.shape[0],k.shape[1],h_kv,d_head).transpose(1,2)
    v = v.reshape(v.shape[0],v.shape[1],h_kv,d_head).transpose(1,2)

    k = torch.repeat_interleave(k, repeats=groups, dim=1)
    v = torch.repeat_interleave(v, repeats=groups, dim=1)
    k_t = k.transpose(-2,-1)

    q_k_t = q @ k_t
    scaled_dk = d_head**(0.5)

    q_k_t_scaled = q_k_t/scaled_dk
    
    if causal:
        s = q.shape[-2]
        mask = torch.triu(
            torch.ones(s,s,dtype=torch.bool), diagonal = 1
        )
        q_k_t_scaled = q_k_t_scaled.masked_fill(mask,float('-inf'))

    # b,h,s_q,s_k @ b,h,s_v,d_head -> b,h,s,d_head
    h_i = F.softmax(q_k_t_scaled, dim = -1) @ v

    # need to transpose so that we can multiply by wo
    # b,h,s,d_head 
    h_i = h_i.transpose(1,2)

    hi_reshape = h_i.reshape(
        h_i.shape[0],
        h_i.shape[1],
        d_model
    )

    y = hi_reshape @ w_o

    return y

    
    
