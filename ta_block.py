# importing necessary modules
import torch
import torch.nn as nn
from attention import MultiHeadAttention


# class for single Transformer Block
class TransformerBlock(nn.Module):

    def __init__(self,
                 hidden_size,
                 num_heads=8,
                 activation="relu",
                 dropout=0.1,
                 expansion_factor=2
                 ):

        """
        Single Transformer Block used in encoder and decoder with MHA and Feed Forward Layers

        Arguments:
            hidden_size : Embedding Dimension
            num_heads : No. of Attention Heads
            activation : Activation used between feed forward. eg., relu, gelu ...
            dropout : Probablities for Dropout Layer in between Layers. (To Avoid Overfitting). default is None
            expansion_factor : The factor that determines the output dimension of the feed forward layer
        """

        super(TransformerBlock, self).__init__()

        # self.input_size = input_size
        self.embedding_dim = hidden_size
        self.n_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        self.attention = MultiHeadAttention(embedding_dim=hidden_size, n_heads=num_heads)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)

        print(self.activation)

        if self.activation == "relu":

            activation = nn.ReLU

        elif self.activation == "gelu":

            activation = nn.GELU

        # the FeedForward layer
        ff_layers = [
            nn.Linear(self.embedding_dim, expansion_factor * self.embedding_dim),  # e.g: 512x(4*512) -> (512, 2048) d_model x d_ff
            activation(), # Applying Actiavation
           nn.Linear(self.embedding_dim * expansion_factor, self.embedding_dim),  # e.g: 4*512)x512 -> (2048, 512) d_ff x d_model
            ]

        if self.dropout is not None:
            self.dropout = nn.Dropout(dropout)
            ff_layers.append(nn.Dropout(dropout))

        self.feed_forward = nn.Sequential(
            *ff_layers
            )


    def forward(self, query, key, value, mask=None):

        """
        Forward Pass for the Transformer Block
        
        Inputs:

            query : tensor of shape (batch_size x sequence length x embedding dims)
            key : tensor of shape (batch_size x sequence length x embedding dims)
            value : tensor of shape (batch_size x sequence length x embedding dims)

            For tasks like sentence embedding, sentiment analysis, classification task query, key, value matricies will be the same
                (Attention will be perform with itself, without mask).
            For tasks like text generation task, query will be input sequence, key and value will be target sequence.
                (Attention will be performed between query and target sequence. with casual attention)

        Returns:

            Tensor which is a result of attention between given q, k, v matrix which passed through feed forward layer of shape (batch_size, seq_len, emedding_dim) which is given to the next block
        """

        # first, pass the key, query and value through the multi head attention layer
        attention = self.attention(query, key, value, mask=mask)  # e.g.: 32x10x512

        # then add the residual connection
        attention_out = attention + query

        """
        The reason are we adding `query` matrix because in the encoder part the input itself is splited into q, k, v so these matrix are same as other,
        but when it comes to decoder, the query matrix comes from masked multi head attention with casual masking performed on decoder input. Therefore,
        In Encoder, q, k, v matricies is same.
        In Decoder k and v comes from encoder output and q from MMHA on decoder's input
        """

        if self.dropout is not None:
            # after that we normalize and use dropout
            attention_drop_normed = self.dropout(self.layer_norm(attention_out))  # e.g.: 32x10x512
        
        else:
            # only layer normalization is performed
            attention_drop_normed = self.layer_norm(attention_out)

        # Feed-Forwar Network
        fc_out = self.feed_forward(attention_drop_normed)  # e.g.:32x10x512 -> #32x10x2048 -> 32x10x512

        # Residual connection
        fc_out = fc_out + attention_drop_normed  # e.g.: 32x10x512

        if self.dropout is not None:
            # dropout + normalization
            fc_norm = self.dropout(self.layer_norm(fc_out))  # e.g.: 32x10x512

        else:
            # just normalization without dropout
            fc_norm = self.layer_norm(fc_out)

        return fc_norm
