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