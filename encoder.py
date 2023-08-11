# import required modules
from math import sin, cos, sqrt, log
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from embedding_utils import EmbeddingLayer, PositionalEncoding
from ta_block import TransformerBlock

# class for Encoder Block in Transformer
class Encoder(nn.Module):

    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 embedding_dim=512,
                 num_blocks=4,
                 activation="relu",
                 expansion_factor=4,
                 num_heads=8,
                 dropout=None
                 ):

        """
        The Encoder part of the Transformer architecture.

        Arguments:
            vocab_size : Vocabulary Size
            embedding_dim :  Dimension to Represent words sequence (Feature for a single word). eg., 256, 512, etc ...
            num_blocks : Number of Transformer block
            activation : Activation to use inbetween feed forward layer. default is `relu`
            expansion_factor : The factor that determines the output dimension of the feed forward layer
            num_heads : Number of Attention Heads
            dropout : Percentage for Droping out the Layers. default is None
        """

        super(Encoder, self).__init__()


        # define the embedding: (vocabulary size x embedding dimension)
        self.embedding = EmbeddingLayer(vocab_size, embedding_dim)

        # define the positional encoding: (max_len x embedding dimension)
        self.pos_emb = PositionalEncoding(max_seq_len, embedding_dim, dropout=dropout)

        stack_them_up = lambda block, n_block: nn.ModuleList([copy.deepcopy(block) for _ in range(n_block)])

        self.transformer_blocks = stack_them_up(TransformerBlock(hidden_size=embedding_dim, activation=activation, num_heads=num_heads, dropout=dropout, expansion_factor=expansion_factor), num_blocks) # Sequentially applies the blocks of the Transformer network

        """
        self.transformer_blocks = nn.Sequential(
                *[TransformerBlock(
                    hidden_size=embedding_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    expansion_factor=expansion_factor
                    ) for _ in range(num_blocks)
                    ]
            )
        """

        # self.output = nn.Linear(input_size, output_size)

    def forward(self, x):

        """
        Forward Pass in Encoder part of transformer
        Inputs:
            x : sequence of tokenized words in batch for parallelism with shape of [batch_size, seq_len]. Note : Here seq_len is fixed len with padded tokens
        Outputs:
            Encoder Representation for the given sequence of words
        """

        # Get the Embeddings for the Sequence
        embedded = self.embedding(x)
    
        # Add Postional Encoding for x
        out = self.pos_emb(embedded)

        # forward pass through transformer blocks
        for block in self.transformer_blocks:

            out = block(out, out, out)

        # Expected Shape - [batch_size, seq_len, embedding_dim]
        return out
