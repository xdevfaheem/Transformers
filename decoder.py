# importing necessary module
import copy
import torch
import torch.nn as nn
from embedding_utils import PositionalEncoding
from decoder_layer import DecoderLayer

# class for single decoder block in transformer architecture
class Decoder(nn.Module):

    def __init__(self,
                 target_vocab_size,
                 max_seq_len,
                 embedding_dim=512,
                 num_blocks=4,
                 activation="relu",
                 expansion_factor=4,
                 num_heads=8,
                 dropout=None
                 ):

        """
        The Decoder part of the Transformer architecture
        Arguments:
            target_vocab_size : Target Vocab Size for Final Projection
            max_seq_len : Maximum Length of the Sequence/words
            embedding_dim : Dimensionality for Embedding
            num_blocks : Number of Decoder Blocks
            activation : Activation to use inbetween feed forward layer. default is `relu`
            expansion_factor : decides the projection of inbetween neurons in feed forward layer
            num_heads : Number of Attention Heads
            droput : percentage of layers to dropout to prevent overfitting and for a stable training. default is None
        """


        super(Decoder, self).__init__()

        self.hidden_size = embedding_dim
        self.n_blocks = num_blocks
        self.dropout = dropout

        self.decoder_embedding = nn.Embedding(target_vocab_size, self.hidden_size)
        self.position_encoder = PositionalEncoding(max_seq_len, self.hidden_size)

        if self.dropout is not None:

            self.dropout = nn.Dropout(dropout)

        stack_them_up = lambda block, n_block: nn.ModuleList([copy.deepcopy(block) for _ in range(n_block)]) # funky name for a function isn't it? :)

        self.decoder_layers = stack_them_up(DecoderLayer(embedding_dim=self.hidden_size, activation=activation, num_heads=num_heads, expansion_factor=expansion_factor, dropout=dropout), num_blocks)

  
    def forward(self, encoder_output, x, mask):

        """
        Forward Pass through Decoder Block
        
        Inputs:
            encoder_output : output from the encoder block (encoder's representation of encoder's input)
            x : decoder's input
        Returns:
            Final probablic distribution over target vocabulary
        """

        if self.dropout is not None:

            out = self.dropout(self.position_encoder(self.decoder_embedding(x)))  # 32x10x512

        else:

            out = self.position_encoder(self.decoder_embedding(x))


        for block in self.decoder_layers:

            out = block(encoder_output, encoder_output, out, mask)

        return out
