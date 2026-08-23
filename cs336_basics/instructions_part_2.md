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



Problem (positionwise_feedforward): Implement the position-wise feed-forward network (2
points)
Deliverable: Implement the SwiGLU feed-forward network, composed of a SiLU activation
function and a GLU.
Note: in this particular case, you should feel free to use torch.sigmoid in your implementation
for numerical stability.
You should set 𝑑ff to approximately 8
3 × 𝑑model in your implementation, while ensuring that the
dimensionality of the inner feed-forward layer is a multiple of 64 to make good use of your
hardware. To test your implementation against our provided tests, you will need to implement the
test adapter at [adapters.run_swiglu] . Then, run uv run pytest -k test_swiglu to test your
implementation.

 
#Rope 
3.4.3 Relative Positional Embeddings
To inject positional information into the model, we will implement Rotary Position Embeddings
[J. Su et al., 2021], often called RoPE. For a given query token 𝑞(𝑖) = 𝑊 𝑞𝑥(𝑖) ∈ ℝ𝑑 at token position 𝑖, we
will apply a pairwise rotation matrix 𝑅𝑖, giving us 𝑞′(𝑖) = 𝑅𝑖𝑞(𝑖) = 𝑅𝑖𝑊 𝑞𝑥(𝑖). Here, 𝑅𝑖 will rotate pairs of
embedding elements 𝑞(𝑖)
2𝑘−1:2𝑘 as 2d vectors by the angle 𝜃𝑖,𝑘 =
𝑖
Θ(2𝑘−2)/𝑑 for 𝑘 ∈ {1, …, 𝑑/2} and some
constant Θ. Thus, we can consider 𝑅𝑖 to be a block-diagonal matrix of size 𝑑 × 𝑑, with blocks 𝑅𝑖
𝑘 for 𝑘 ∈
{1, …,
𝑑
2 }, with
𝑅𝑖
𝑘 = (cos(𝜃𝑖,𝑘)
sin(𝜃𝑖,𝑘)
− sin(𝜃𝑖,𝑘)
cos(𝜃𝑖,𝑘) ) (8)
Thus we get the full rotation matrix
𝑅𝑖
where 0s represent 2 × 2 zero matrices. While one could construct the full 𝑑 × 𝑑 matrix, a good solution
should use the properties of this matrix to implement the transformation more efficiently. Since we only
care about the relative rotation of tokens within a given sequence, we can reuse the values we compute for
cos(𝜃𝑖,𝑘) and sin(𝜃𝑖,𝑘) across layers, and different batches. If you would like to optimize it, you may use a
single RoPE module referenced by all layers, and it can have a 2d pre-computed buffer of sin and cos
values created during init with self.register_buffer(persistent=False), instead of an nn.Parameter
(because we do not want to learn these fixed cosine and sine values). The exact same rotation process we
did for our 𝑞(𝑖) is then done for 𝑘(𝑗), rotating by the corresponding 𝑅𝑗. Notice that this layer has no
learnable parameters.
Problem (rope): Implement RoPE (2 points)
Deliverable: Implement a class RotaryPositionalEmbedding that applies RoPE to the input
tensor.
The following interface is recommended:
def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) Construct the
RoPE module and create buffers if needed.
theta: float Θ value for the RoPE
d_k: int dimension of query and key vectors
max_seq_len: int Maximum sequence length that will be input
device: torch.device | None = None Device to store the buffer on
def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor Process
an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. Note
that you should tolerate 𝑥 with an arbitrary number of batch dimensions. You should assume
that the token positions are a tensor of shape (..., seq_len) specifying the token positions of
𝑥 along the sequence dimension.
You should use the token positions to slice your (possibly precomputed) cos and sin tensors along
the sequence dimension.
To test your implementation, complete [adapters.run_rope] and make sure it passes uv run
pytest -k test_rope

#softmax 
Problem (softmax): Implement softmax (1 point)
Deliverable: Write a function to apply the softmax operation on a tensor. Your function should
take two parameters: a tensor and a dimension 𝑖, and apply softmax to the 𝑖-th dimension of the
input tensor. The output tensor should have the same shape as the input tensor, but its 𝑖-th
dimension will now have a normalized probability distribution. Use the trick of subtracting the
maximum value in the 𝑖-th dimension from all elements of the 𝑖-th dimension to avoid numerical
stability issues.
To test your implementation, complete [adapters.run_softmax] and make sure it passes uv run
pytest -k test_softmax_matches_pytorch.

#scaled dot product attention

Problem (scaled_dot_product_attention): Implement scaled dot-product attention (5
points)
Deliverable: Implement the scaled dot-product attention function. Your implementation should
handle keys and queries of shape (batch_size, ..., seq_len, d_k) and values of shape
(batch_size, ..., seq_len, d_v), where ... represents any number of other batch-like
dimensions (if provided). The implementation should return an output with the shape
(batch_size, ..., seq_len, d_v). See Section 3.2 for a discussion on batch-like dimensions.
Your implementation should also support an optional user-provided boolean mask of shape
(seq_len, seq_len). The attention probabilities of positions with a mask value of True should
collectively sum to 1, and the attention probabilities of positions with a mask value of False
should be zero.

To test your implementation against our provided tests, you will need to implement the test
adapter at [adapters.run_scaled_dot_product_attention] . uv run pytest -k test_scaled_dot_product_attention tests your implementation on third-order input tensors, while
uv run pytest -k test_4d_scaled_dot_product_attention tests your implementation on fourth-
order input tensors.

#multi head self attention
Problem (multihead_self_attention): Implement causal multi-head self-attention (5 points)
As a stretch goal, try combining the key, query, and value projections into a single weight matrix so you only need a
single matrix multiply.
Deliverable: Implement causal multi-head self-attention as a torch.nn.Module. Your
implementation should accept (at least) the following parameters:
d_model: int Dimensionality of the Transformer block inputs.
num_heads: int Number of heads to use in multi-head self-attention.
Following A. Vaswani et al. [8], set 𝑑𝑘 = 𝑑𝑣
𝑑model
=
ℎ . To test your implementation against our
provided tests, implement the test adapter at [adapters.run_multihead_self_attention
