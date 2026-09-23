import torch
import torch.nn.functional as F

def multi_query_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_query_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns an attention tensor with the same shape as hidden_states.
    """
    X = hidden_states
    # the shape of x is (B,S,d_model) so i need to get the model shape out 
    d_model = X.shape[-1]

    # generating Q,K,V by multiplying with X = (B,S,d_model) to Q,K,V belongs to (d_model,d_model) so when we multiply we get (B,S,d_model)
    Q = X @ w_q
    K = X @ w_k
    V = X @ w_v

    # now we need to calculate d_k by using d_model and using number of heads 
    d_head = d_model//num_query_heads

    # getting query into shape Q = B,S,d_model -> B,S,H,d_head -> B,H,S,d_head
    q_heads = Q.reshape(Q.shape[0],Q.shape[1],num_query_heads,d_head).transpose(1,2)

    # so one question I have here should I build a single head thats it so num_key and value heads will  be 1
    num_key_value_heads = 1
    k_heads = K.reshape(K.shape[0],K.shape[1],num_key_value_heads,d_head).transpose(1,2)
    v_heads = V.reshape(V.shape[0],V.shape[1],num_key_value_heads,d_head).transpose(1,2)
    # here we need to transpose k, for easier mat mul -> b,h,s_q,d_head @ b,h,d_head,s_k -> b,h,s_q,s_k
    k_t = k_heads.transpose(-2,-1)

    q_k_t = q_heads @ k_t
    scaled_value = (q_k_t) / (d_head**(0.5))

    if causal:
        # we need to take the s size for masking B,h,s,d
        s = q_heads.shape[-2]
        # building the mask -> build a matrix first with ones with the shape of s,s and set upper diagonal values as inf/0 
        mask = torch.triu(torch.ones(s,s,dtype=torch.bool), diagonal=1)
        # apply the maks with soft max 
        scaled_value = scaled_value.masked_fill(mask,float("-inf"))
    # b,h,s_q,s_k @ b,h,s_v,d_head -> b,h,s,d_head
    h_i = F.softmax(scaled_value,dim=-1) @ v_heads

    # wo -> d_model,d_model , b,h,s,d_head -> b,s,d_model
    h_i = h_i.transpose(1,2)
    h_concat = h_i.reshape(
        h_i.shape[0],
        h_i.shape[1],
        d_model
    )

    y = h_concat @ w_o

    return y
        

    
    
