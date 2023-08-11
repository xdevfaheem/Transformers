# importing necessary module
import copy
import torch
import torch.nn as nn
from activation import Softmax
from decoder import Decoder
from encoder import Encoder


# class for Complete Transformers Architecture (TA)
class TransformerNet(nn.Module):

    def __init__(self,
                 d_model,
                 inp_vocab_size,
                 target_vocab_size,
                 input_max_len,
                 target_max_len,
                 n_blocks=6,
                 activation="relu",
                 expansion_factor=4,
                 n_heads=8,
                 dropout_size=None):

        """
        Complete Transformer Neural Network
        
        Arguments:
            d_model : Embedding Dimension
            inp_vocab_size : Vocabulary Size of the Input/Source
            target_vocab_size : Vocabulary Size of Target for Projection
            input_max_len : Maximum Sequence Length of the input
            target_max_len : Maximum Sequence Length of the Target
            n_blocks : Number of Encoder/Decoder block for the Model
            activation : Activation to use inbetween feed forward layer. default is `relu`
            expansion_factor : Determine the Inner Dimension of the feed forward layer
            n_heads : Number of Attention Heads
            dropout_size : percentage of the layer to drop inbetween the layers to prevent overfitting and stablize the training
        """

        super(TransformerNet, self).__init__()

        # Single Encoder Block
        self.encoder = Encoder(
            inp_vocab_size,
            input_max_len,
            embedding_dim=d_model,
            num_blocks=n_blocks,
            expansion_factor=expansion_factor,
            activation=activation,
            num_heads=n_heads,
            dropout=dropout_size
        )

        # Single Decoder Block
        self.decoder = Decoder(
            target_vocab_size,
            target_max_len,
            embedding_dim=d_model,
            num_blocks=n_blocks,
            activation=activation,
            expansion_factor=expansion_factor,
            num_heads=n_heads,
            dropout=dropout_size
        )

        self.softmax = Softmax(axis=-1, keepdim=True)

        # Final Fully Connected Layer to project the distribution over target vocabulary
        self.fc_out = nn.Linear(d_model, target_vocab_size)

    
    # function to create mask for MMHA
    def _create_target_mask(self, tg):
        batch_size, trg_len = tg.shape

        # mask out the lower diagnol of the matrix for casual attention
        trg_mask = torch.tril(torch.ones((batch_size, 1, trg_len, trg_len))).type(torch.bool)
        return trg_mask

    
    def forward(self, input, target):

        """
        Forward Pass through Transformer Architecture
        
        Inputs:
            input : Input to the Encoder
            target : Input to the Decoder
            eg., for task like Neural Machine Translation (NMT), the input will be source language and target will be language you want to translate to.
        Returns:
            Probablic Distribution Over Entire Target Vocabulary
        """

        trg_mask = self._create_target_mask(target) # mask used for casual attention

        # Forward Pass through the Encoder Layer
        enc_out = self.encoder(input)

        # Forward pass through Decoder Layer
        outputs = self.decoder(enc_out, target, trg_mask)

        # Softmax over finally linear projected logits along the final dimension
        output = self.softmax(self.fc_out(outputs))

        return output
