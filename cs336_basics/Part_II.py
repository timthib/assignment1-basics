import torch


class Linear(torch.nn.Module) :

    def __init__(self, in_features, out_features, device=None, dtype=None) :
        """Construct a linear transformation module. 
        This function should accept the following parameters:
    in_features: int final dimension of the input
    out_features: int final dimension of the output
    device: torch.device | None = None Device to store the parameters on ("cpu", "cuda", None = default = cpu)
    dtype: torch.dtype | None = None Data type of the parameters (ft32, ft16, bf16,...)

    """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weights = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(out_features,in_features, device=device, dtype=dtype)))

    def forward(self, x: torch.Tensor) -> torch.Tensor :
        """
        Apply the linear transformation to the input.
        """
        return x @ self.weights.T


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None) :
        """Construct an
        embedding module. This function should accept the following parameters:
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors, i.e., 𝑑model
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device 
        self.dtype = dtype 
        self.weights = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)))



    def forward(self, token_ids: torch.Tensor) -> torch.Tensor : 
        """ Lookup the embedding vectors for the given token IDs.
        self.weigths is a vocab_size x dim_model matrix 
        token ids is a batch_size x sequence_length matrix composed of index of vocabulary in [0, vocab_size]
        self.weights[token_ids] creates a tensor shaped like token_ids replacing each index by its corrsponding row in self.weights 
        ending with batch * sequence_length * dim_model

        """
        return self.weights[token_ids]



class rmsnorm(torch.nn.Module) :

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        RMSNorm module. This function should accept the following parameters:
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weights = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(d_model,device=device, dtype=dtype)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor :
        """
        Process an input tensor of shape
        (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        """
        in_type = x.dtype
        x = x.to(torch.float32)
        mean_on_token = (x ** 2).mean(dim=-1,keepdim=True) 
        #square each entry, compute mean over token dim (d=-1 = d_model)
        #keepdim => output is (batch,seq,1) instead of (batch,seq), strange but need it for next step
        rms = torch.sqrt(mean_on_token + self.eps) #shape batch,seq,1
        result = x/rms # shape batch,seq,d_model, divide each of the d_model values at the batch,seq position of x by the 1 value of rms at the batch,seq position
        result = result * self.weights # batch, seq, d_model * d_model: element wise multiplication on each token vector
        return result.to(in_type)
    

class positionwise_feedforward(torch.nn.Module): 

    def __init__(self,d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(self.d_ff,self.d_model)))
        self.w2 = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(self.d_model,self.d_ff)))
        self.w3 = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(self.d_ff,self.d_model)))

    def forward(self,x):
        inter = x @ self.w1.T
        silu = (inter) * torch.sigmoid(inter)
        return (silu * (x @ self.w3.T)) @ self.w2.T
    

class RotaryPositionalEmbedding(torch.nn.Module) : 

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) :
        """
        Construct the
        RoPE module and create buffers if needed.
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be input
        device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device 
        self.den = torch.tensor([1/self.theta**((2*k-2)/self.d_k) for k in range(1,self.d_k//2 +1)]) #shape: d_k/2
        positions = torch.arange(self.max_seq_len) #shape: max_seq_len
        self.angles = torch.outer(positions, self.den)   #shape: max_seq_len * d_k/2    #torch.stack([i*self.den for i in range(max_seq_len)]) #use stack to make a tensor of tensors
        
        self.register_buffer("cos_cached", torch.cos(self.angles), persistent = False) #shape: max_seq_len * d_k/2
        self.register_buffer("sin_cached", torch.sin(self.angles), persistent = False) #shape: max_seq_len * d_k/2
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor : 
        """
        Process
        an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. Note
        that you should tolerate 𝑥 with an arbitrary number of batch dimensions. You should assume
        that the token positions are a tensor of shape (..., seq_len) specifying the token positions of
        𝑥 along the sequence dimension.
        You should use the token positions to slice your (possibly precomputed) cos and sin tensors along
        the sequence dimension.
        """
        
        #token position = [0,1,2] for a 3 rows input 

        x_even = x[...,::2] #shape ...,seq_len, d_k/2
        x_odd = x[...,1::2] #values of x on the odd indices 1,3,5,..., use '...' to ensure we select odd positions on the last dimention  
        cos = self.cos_cached[token_positions] #shape: ..., seq_len, d_k/2
        sin = self.sin_cached[token_positions] #shpae: ..., seq_len, d_k/2
        res1 = x_even*cos - x_odd*sin  # x_odd and cos have same shape so its only element wise multiplication
        res2 = x_even*sin + x_odd*cos 
        out = torch.stack([res1,res2], dim=-1) #res1 = [a,b], res2 = [c,d] => out = [[a,c], [b,d]]
        out = out.flatten(start_dim=-2) # out = [a,c,b,d]
        return out



def softmax(tens, dim_i): 
    """
    Write a function to apply the softmax operation on a tensor. Your function should
    take two parameters: a tensor and a dimension 𝑖, and apply softmax to the 𝑖-th dimension of the
    input tensor. The output tensor should have the same shape as the input tensor, but its 𝑖-th
    dimension will now have a normalized probability distribution. Use the trick of subtracting the
    maximum value in the 𝑖-th dimension from all elements of the 𝑖-th dimension to avoid numerical
    stability issues.
    """
    mx = torch.amax(tens, dim = dim_i, keepdim=True) #torch.max returns (values, indices), torch.amax only return values  tensor([[0.2371, 0.5778, 0.8317]]), 1 value per row of the dim_i
    vect = tens-mx  #for each row of the given dim, substract the max of the row 
    s = torch.sum(torch.exp(vect), dim = dim_i, keepdim=True) #sum rows on dim_i
    vect = torch.exp(vect)/s
    return vect

def scaled_dot_product_attention(Q,K,V,M=None):
    if M is not None: 
        M = torch.where(M,0.0,float('-inf')) #torch.where(condition,a,b) pick a if condition is True, b when condition is false
        S = Q @ K.transpose(-2,-1) + M 
    else : 
        S = Q @ K.transpose(-2,-1) #Q and K: batch,...,seq len, d_k, so transpose (-2-1) => batch,..., d_k,seq len 
        #so product is batch,..., seq_len_q, d_k @ batch,..., d_k,seq_len_k
    A = softmax(S/torch.sqrt(torch.tensor(K.shape[-1])),-1) @ V
    return A

class multihead_self_attention(torch.nn.Module):

    def __init__(self,d_model, num_heads, max_seq_len,theta, Wq, Wk, Wv, Wo):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.Wq = Wq # d_model  * d_k
        self.Wk = Wk # d_model  * d_k
        self.Wv = Wv # d_model  * d_v
        self.Wo = Wo # d_model  * d_model
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rope = RotaryPositionalEmbedding(self.theta, self.Wk.shape[-1]//num_heads, self.max_seq_len)

    def forward(self,x,token_positions):
        heads = []
        seq_len = x.shape[-2]
        m = torch.tril(torch.ones(seq_len,seq_len,dtype=torch.bool)) #lower = 1, upper = 0, shape = seq_len * seq_len
        Q = x @ self.Wq.T
        K = x @ self.Wk.T
        V = x @ self.Wv.T

        for h in range(self.num_heads):
            range_head = h*self.d_model//self.num_heads,(h+1)*self.d_model//self.num_heads
            
            if token_positions is None : 
                token_positions = torch.arange(seq_len) #shape ?
            
            q = Q[...,range_head[0]:range_head[1]] #shape batch, ..., seq_len, d_model
            q = self.rope.forward(q,token_positions)
            
            k = K[...,range_head[0]:range_head[1]]
            k = self.rope.forward(k,token_positions)
            
            v = V[...,range_head[0]:range_head[1]]
            
            a = scaled_dot_product_attention(q,k,v,m)
            heads.append(a)

        multi_h = torch.cat(heads,dim=-1)     
        
        return  multi_h @ self.Wo.T
    


