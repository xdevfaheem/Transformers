# import necessary modules
import torch
import torch.nn as nn
from attention import MultiHeadAttention
from ta_block import TransformerBlock


# class for Single Decoder Layer in Transformers
class DecoderLayer(nn.Module):

    def __init__(
        self,
        embedding_dim=512,
        num_heads=8,
        activation="relu",
        expansion_factor=4,
        dropout=None
    ):

        """
        Single Decoder Layer in Transformer Architechture (TA) which consist of transformer block used in both encoder and decoder and
        a Masked Multi Head Attention with casual masking.

        Casual Masking:
            In Self Attention, query and key matricies are same without masking, one word representation in query can see every other word representation in key matrix. this way,
            a word can see every other word within a same sentence but, In casual Atention, Every word in sentence can only see past tokens to predict next token. Here, casual masking
            is used to prevent current token from seeing the future token.

            This is the Reason Why Encoder which learns language representation, is used for question answering, as a MLMs, for sentiment analysis, sentence embedding ... which require contextual awareness.
            and Decoder which learns to predict to next word/token based on past tokens, is used for tasks like Text generation, Next Sentence prediction (NSP)

        Args:
            embedding_dim : Enbedding Dimension
            num_heads : Number of heads
            activation : Activation to use inbetween feed forward layer. default is `relu`
            expansion_factor : decides the inner dimension of feed forward layer
            dropout : percentage of layers to drop. default is None
        """

        super(DecoderLayer, self).__init__()

        self.hidden_size = embedding_dim # hidden size of the decoder
        self.dropout = dropout # dropout percentage

        self.attention = MultiHeadAttention(embedding_dim, num_heads) # Initialising Multi Head Attention

        self.layer_norm = nn.LayerNorm(embedding_dim) # Layer Normalization for stablizing the tensor's between layers

        if self.dropout is not None:

            self.dropout = nn.Dropout(dropout) # Initialising Dropout

        self.transformer_block = TransformerBlock(embedding_dim, activation=activation, num_heads=num_heads, expansion_factor=expansion_factor, dropout=self.dropout) # Transformer block used in both encoder and decoder

    
    def forward(self, x, key, value, mask):

        """
        Forward Pass to Decoder Layer
        
        Inputs:
            x - Decoder's Input
            key -  Encoder Representation of Encoder's input (Encoder Output)
            value - Encoder Representation (This is also Encoder's Output)
        Returns:
            TransfomerBlock's Output between Encoder Representation of Encoder's Input (q, k) and Masked Multi Head Attention within Decoder's Input (Decoder Reprensentation of Decoder's Input)
        """

        # MMHA
        masked_multi_head_attention = self.attention(x, x, x, mask=mask) # Performing casual attention on decoder's input

        if self.dropout is not None:
            # residual connection and normalization with dropout at last
            query = self.dropout(self.layer_norm(masked_multi_head_attention + x))

        else:
            # residual connection and normalization without dropout
            query = self.layer_norm(masked_multi_head_attention + x)

        # forward pass to Transformer Block which consist of multi head attention between encoder output q, k and decoder's output (value)
        # and feed forward network in addition to that add & norm inbetween layer to prevent overfitting and stable training als, some dropouts if you opt in :)
        decoder_attention = self.transformer_block(query=query, key=key, value=value) # we are not passing the mask as self attenttion will be performed among them not casual attention

        return decoder_attention
