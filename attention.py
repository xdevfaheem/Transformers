# Import Necessary Modeules
from math import sin, cos, sqrt, log
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# class for Multi Head Attention Module
class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim=512, n_heads=8):

        """

        Multi Head Attention

        Multi-Head Attention is a module for attention mechanisms which runs through an attention mechanism several times in parallel.
        The key idea behind MHA is that, different attention heads can capture different types of relationships within the input sequence.
        By using Multiple Heads, the model can attend to various aspects of the input simultaneously, making it more robust and capable of learning complex relationships.

        Arguments:
            embedding_dim : Embedding Dimension of the input Sequence
            n_heads : Number of Attention Heads to run in parallel

        """

        super(MultiHeadAttention, self).__init__()

        assert embedding_dim % n_heads == 0, "Embedding  Dimension divided by no. of heads should give no remainder. so that it can be equally splitted"

        self.embedding_dim = embedding_dim # Embedding Dimension of the model
        self.n_heads = n_heads # Number of Attention Heads. eg., 8 by default
        self.head_size = int(embedding_dim // n_heads) # Embedding Dimension for a single head
        self.softmax = Softmax(axis=-1, keepdim=True) # Custom Softmax layer

        # Weighted Matricies
        self.query = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False) # weighted matricies to transform/project the query matrix to perform self attention
        self.key = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)  # weighted matricies to transform/project the key to perfom self attention
        self.value = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)  # weighted matricies to transform/project the value matrix

        self.final_fc_layer = nn.Linear(self.head_size * self.n_heads, self.embedding_dim) # Final Layer for projecting the q, k, v matricies into a single tensor of shape(embedding_dim, embedding_dim)

    def _scaled_dot_product_attention(self, query, key, value, mask=None, scale=True):

        """
        Dot product self-attention.
        Args:
            query : array of query representations with shape (batch_size, q_len, heads, head)
            key : array of key representations with shape (batch_size, q_len, heads, head)
            value : array of value representations with shape (batch_size, q_len, heads, head)
            mask : attention-mask, gates attention
            scale : whether to scale the dot product of the query and transposed key

        Returns:
            Self-attention array for q, k, v arrays. (L_q by d)
        """

        assert query.shape[-1] == key.shape[-1] == value.shape[-1], "Embedding dimensions of q, k, v aren't all the same"

        if scale:
            depth = query.shape[-1] # Scaling factor for the dot product attention
        else:
            depth = 1

        # query shape: batch_size, n_heads, q_len, head_size. e.g: (32x8x10x64)
        # key shape: batch_size, n_heads, k_len, head_size. e.g: (32x8x10x64). k_transposed shape: batch_size, n_heads, head_size, k_len. eg., (32, 8, 64, 10)
        # product shape should be: batch_size, heads, q_len, v_len, e.g: (32x8x10x10)
        dots = torch.matmul(query,key.transpose(2,3))

        if mask is not None:

            """
            if mask is given, fill the lower triangle with dot product values. default the mask is None as mask is not used in self attention.
            it can be passed when the the this class used in decoder where we perform casual attention which needs the dots matrix to be masked.
            to prevent the current word from seeing upcoming/future words.
            """

            dots = torch.where(mask, dots, torch.full_like(dots, -1e13))

        scores = self.softmax(dots / torch.sqrt(torch.tensor([self.head_size * 1.0]).to(dots))) # perform softmax operation dot product scaled by the scaling factor

        # scores shape: batch_size, heads, q_len, v_len, e.g: (32x8x10x10)
        # value shape: batch_size, heads, v_len, head_size, e.g: (32x8x10x64)
        # output: batch_size, heads, q_len, head_size, e.g: (32x8x10x64)
        weights = torch.matmul(scores, value)
        weights.permute(0, 2, 1, 3).contiguous() # Swapping the second and first fimension of the weights matrix. resulting matrix has a shape of [batch_size, v_Len, heads, head_size]

        return weights, scores

    def forward(self, query, key, value, mask=None, return_attention=False):

        """
        Forward pass of the Multi-Head Attention module.

        Inputs:
            query (Tensor): The input query tensor of shape (batch_size, seq_len_query, input_dim).
            key (Tensor): The input key tensor of shape (batch_size, seq_len_key, input_dim).
            value (Tensor): The input value tensor of shape (batch_size, seq_len_value, input_dim).
            mask (Tensor, optional): Mask tensor to mask future positions in casual attention. default is None. Note: mask is not used in self-attention

        Returns:
            output (Tensor): The output tensor after multi-head attention, of shape (batch_size, seq_len_query, input_dim).
            attention_weights (Tensor): Attention weights of shape (batch_size, num_heads, seq_len_query, seq_len_key). it is optional.
        """

        # Input of size: batch_size x sequence length x embedding dims
        batch_size = query.size(0)
        k_len, q_len, v_len = key.size(1), query.size(1), value.size(1)


        # project the queries, keys and values by their respective weight matrices
        key = self.key(key)  # [batch_size, seq_len, embedding_dim]
        query = self.query(query)  # [batch_size, seq_len, embedding_dim]
        value = self.value(value)  # [batch_size, seq_len, embedding_dim]


        # reshape from (batch_size x seq_len x embed_size) -> (batch_size x seq_len x n_heads x head_size)
        # example: from (32x10x512) -> (32x10x8x64)
        query = query.view(batch_size, q_len, self.n_heads, self.head_size).permute(0, 2, 1, 3) # [batch_size, seqLen, n_head, head_size] -> [batch_size, n_head, seqLen, head_size]
        key = key.view(batch_size, k_len, self.n_heads, self.head_size).permute(0, 2, 1, 3) # [batch_size, seqLen, n_head, head_size] -> [batch_size, n_head, seqLen, head_size]
        value = value.view(batch_size, v_len, self.n_heads, self.head_size).permute(0, 2, 1, 3) # [batch_size, seqLen, n_head, head_size] -> [batch_size, n_head, seqLen, head_size]


        weights, attention = self._scaled_dot_product_attention(query, key, value, mask=mask) # batch_size, heads, v_len, head_size,

        output = self.final_fc_layer(weights.view(batch_size, q_len, self.n_heads * self.head_size)) # (batch_size, seq_len, embedding_dims)

        if return_attention:
            return output, attention
        else:
            return output
