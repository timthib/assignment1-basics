Problem (linear): Implementing the linear module (1 point)
Deliverable: Implement a Linear class that inherits from torch.nn.Module and performs a linear
transformation. Your implementation should follow the interface of PyTorch’s built-in nn.Linear
module, except for not having a bias argument or parameter. We recommend the following
interface:
def __init__(self, in_features, out_features, device=None, dtype=None) Construct a linear
transformation module. This function should accept the following parameters:
in_features: int final dimension of the input
out_features: int final dimension of the output
device: torch.device | None = None Device to store the parameters on
dtype: torch.dtype | None = None Data type of the parameters
def forward(self, x: torch.Tensor) -> torch.Tensor Apply the linear transformation to the
input.
Make sure to:
• subclass nn.Module
• call the superclass constructor
• construct and store your parameter as 𝑊 (not 𝑊 ⊤), putting it in an nn.Parameter
• of course, don’t use nn.Linear or nn.functional.linear
For initializations, use the settings from above along with torch.nn.init.trunc_normal_ to
initialize the weights.
To test your Linear module, implement the test adapter at [adapters.run_linear] . The adapter
should load the given weights into your Linear module. You can use Module.load_state_dict for
this purpose. Then, run uv run pytest -k test_linear.

3.3.3 Embedding Module
As discussed above, the first layer of the Transformer is an embedding layer that maps integer token IDs
into a vector space of dimension d_model. We will implement a custom Embedding class that inherits from
torch.nn.Module (so you should not use nn.Embedding). The forward method should select the embedding
vector for each token ID by indexing into an embedding matrix of shape (vocab_size, d_model) using a
torch.LongTensor of token IDs with shape (batch_size, sequence_length).
Problem (embedding): Implement the embedding module (1 point)
Deliverable: Implement the Embedding class that inherits from torch.nn.Module and performs an
embedding lookup. Your implementation should follow the interface of PyTorch’s built-in
nn.Embedding module. We recommend the following interface:


def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None) Construct an
embedding module. This function should accept the following parameters:
num_embeddings: int Size of the vocabulary
embedding_dim: int Dimension of the embedding vectors, i.e., 𝑑model
device: torch.device | None = None Device to store the parameters on
dtype: torch.dtype | None = None Data type of the parameters
def forward(self, token_ids: torch.Tensor) -> torch.Tensor Lookup the embedding vectors
for the given token IDs.
Make sure to:
• subclass nn.Module
• call the superclass constructor
• initialize your embedding matrix as an nn.Parameter
• store the embedding matrix with the d_model being the final dimension
• of course, don’t use nn.Embedding or nn.functional.embedding
Again, use the settings from above for initialization, and use torch.nn.init.trunc_normal_ to
initialize the weights.
To test your implementation, implem




#RMSNorm
Each Transformer block has two sub-layers: a multi-head self-attention mechanism and a position-wise
feed-forward network ([A. Vaswani et al., 2017], section 3.1).
In the original Transformer paper, the model uses a residual connection around each of the two sub-
layers, followed by layer normalization. This architecture is commonly known as the “post-norm”
Transformer, since layer normalization is applied to the sub-layer output. However, a variety of work has
found that moving layer normalization from the output of each sub-layer to the input of each sub-layer
(with an additional layer normalization after the final Transformer block) improves Transformer training
stability [T. Q. Nguyen et al., 2019; R. Xiong et al., 2020] — see Figure 2 for a visual representation of
this “pre-norm” Transformer block. The output of each Transformer block sub-layer is then added to the
sub-layer input via the residual connection (A. Vaswani et al. [8], section 5.4). An intuition for pre-norm
is that there is a clean “residual stream” without any normalization going from the input embeddings to
the final output of the Transformer, which is purported to improve gradient flow. This pre-norm
Transformer is now the standard used in language models today (e.g., GPT-3, LLaMA, PaLM, etc.), so
we will implement this variant. We will walk through each of the components of a pre-norm Transformer
block, implementing them in sequence.
3.4.1 Root Mean Square Layer Normalization
The original Transformer implementation of A. Vaswani et al. [8] uses layer normalization
[J. L. Ba et al., 2016] to normalize activations. Following H. Touvron et al. [12], we will use root mean
square layer normalization (RMSNorm; B. Zhang et al. [13], equation 4) for layer normalization. Given a
vector 𝑎 ∈ ℝ𝑑model of activations, RMSNorm will rescale each activation 𝑎𝑖 as follows:
RMSNorm(𝑎𝑖) =
𝑎𝑖
RMS(𝑎) 𝑔𝑖, (4)
19
where RMS(𝑎) = √1
𝑑model ∑𝑑model
𝑖=1 𝑎2
𝑖 + 𝜀. Here, 𝑔𝑖 is a learnable “gain” parameter (there are d_model such
parameters total), and 𝜀 is a hyperparameter that is often fixed at 1e-5.
You should upcast your input to torch.float32 to prevent overflow when you square the input. Overall,
your forward method should look like:
in_dtype = x.dtype
x = x.to(torch.float32)
# Your code here performing RMSNorm
...
result =
...
# Return the result in the original dtype
return result.to(in_dtype)

Problem (rmsnorm): Root Mean Square Layer Normalization (1 point)
Deliverable: Implement RMSNorm as a torch.nn.Module. We recommend the following interface:
def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None) Construct the
RMSNorm module. This function should accept the following parameters:
d_model: int Hidden dimension of the model
eps: float = 1e-5 Epsilon value for numerical stability
device: torch.device | None = None Device to store the parameters on
dtype: torch.dtype | None = None Data type of the parameters
def forward(self, x: torch.Tensor) -> torch.Tensor Process an input tensor of shape
(batch_size, sequence_length, d_model) and return a tensor of the same shape.
Note: Remember to upcast your input to torch.float32 before performing the normalization
(and later downcast to the original dtype), as described above.
To test your implementation, implement the test adapter at [adapters.run_rmsnorm] . Then, run
uv run pytest -k test_rmsnorm.