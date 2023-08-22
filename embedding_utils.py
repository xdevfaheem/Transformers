# Importing Required Libraries
import torch
import torch.nn as nn


# class for Embedding Layer to project the input sequences to Multi Dimensional Space
class EmbeddingLayer(nn.Module):

    def __init__(self, vocab_size, embedding_dim):

        """
	    Class to transform word sequences to Multi Dimensional Space (Numerical Reprensentation)
	
	    Arguments:
	        vocab_size : Vocablary Size
	        embedding_dim : Dimension to Represent words sequence (Feature for a single word).
	                        eg., 256, 512 (As the Dimension Increases, More Dependencies/Context can be Capture as well need more computation)
	
	    For example, if you have a batch of 64 sequences, each containing 15 words, and the embedding dimension is 512,
	    the output tensor will be of size 64x15x512, where each element in the tensor represents a numerical embedding.
	    """

        super(EmbeddingLayer, self).__init__()
	    
        self.embed_dim = embedding_dim
        self.embed = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
		
        """
        Inputs:
            x : The input word(s) or sequence of words that need to be embedded.
		Returns:
  			Embeddings of the Given sequence with shape (B x S x D)
        """
		
        x = self.embed(x) # This gives a tensor representing the embeddings of the input words.
        embedded = x * torch.sqrt(torch,tensor(self.embed_dim)) # This scaling factor is often used to prevent gradient explosions when training deep networks.
        return embedded # The resulting tensor is the numerical representation of the input in the embedding space.
		

"""
Some Codes to play around with Embeddings which i used:

max_seq_len = 100
embedding_dim = 10
embedder = EmbeddingLayer(max_seq_len, embedding_dim)
x = torch.randint(1, 20, size=(64, ), dtype=torch.int).reshape(8, 8) # batchsize, seq_len
print(x, "\n", x.shape)
embedded = embedder(x)
print()
print(embedded)
print()
print(embedded.shape) # batchsize, seq_len, embedding_dim
"""


# class for Positional Encoding to injects some information to the embedded values about the relative or absolute position of the tokens in the sequence
class PositionalEncoding(nn.Module):


    def __init__(self, max_len, d_model=512, dropout=None, n=10000.0):

        """
        class for Positional Embedding or Positional Encoding in Transfomer Architechture

        This addresses the issue of sequence order and helps the model understand the relative positions of tokens within a sequence.

        In LSTMs, GRUs,  Recurrent Neural networks (RNN), the inputs are fed into the model sequentially. For each timestamps, the input word is fed and the corresponding hidden state is obtained.
        This way, the model learns the relative position of the word within a sequence.

        But in the Transformer architecture, the model processes tokens in parallel using self-attention mechanisms.
        Since self-attention doesn't inherently take into account the position of tokens,
        positional embeddings are added to the input embeddings to provide information
        about the positions of tokens in the sequence.

        Arguments:
            max_len : Maximum Length of the Sequence
            embedding_dim : Dimension of the Embedding, This Must be Same as Embedding vector
            drouput : Dropout Probablity

        """

        super(PositionalEncoding, self).__init__()
        
        self.max_len = max_len
        self.embedding_dim = d_model
		self.dropout = dropout
		
		if self.dropout is not None:
        	self.pos_dropout = nn.Dropout(dropout)
        self.n = n

        positional_encoding = torch.zeros(max_len, d_model)  # Matrix Filled with zeros of shape (max_len, embedding_dim)
        position = torch.arange(0, max_len).unsqueeze(axis=1) # Positions/Index of the Words in a sequence

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(self.n)) / d_model))

        """
        Denominator for Scaling. These exponential values help create a pattern that contributes to the unique encoding for each position
        This has Many Benefits,
        - It control the frequency of oscillation (how quickly the function oscillates as position changes)
        - It Ensures each position has unique and distinctive. Without scaling, the same positional encodings could repeat over and over,
          making it difficult for the model to differentiate positions. (Encoding Relative Positions)
        - The positional encoding is designed to handle sequences of varying lengths.
        """

        """
        for i in position:
            for j in torch.arange(0, embedding_dim, 2)):
                positional_encoding[i, 2*j] = pe_sin(i, j)
                positional_encoding[i, 2*j+1] = pe_cos(i, j)

        You Can use this if you want but it can be done efficiently using vectorized operation, done below
        """

        # Vectorized Operation
        positional_encoding[:, 0::2] = torch.sin(position * div_term) # apply sin functions for every two coloumn of pos_emb matrix starting from 0. This term `position * div_term` has shape (max_seq_len, embedding_dim/2)
        positional_encoding[:, 1::2] = torch.cos(position * div_term) # apply cosine functions for every two coloumn of pos_emb matrix starting from 0.

        self.pe = positional_encoding.unsqueeze(0) # Add Extra Batch Dimension along the first axis
        self.register_buffer('pe', self.pe) # Register Buffer to make it a part of the module's state_dict

    def _pe_sin(self, position, i): # internal sin function
        return torch.sin(position / torch.pow(self.n, ((2 * i) / self.embedding_dim)))

    def _pe_cos(self, position, i): # internal cosine function
        return torch.cos(position / torch.pow(self.n, ((2 * i) / self.embedding_dim)))

    def forward(self, x):

	"""
	Forward Pass through Positional Encoding Layer

	Inputs:
		x : Embedded Sequence
	Returns:
		The Positional Encoding added to Embedded Sequence to provide information
	about the positions of tokens in the sequence.
	"""
		
        # print(x.shape, self.pe.shape)
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)

		if self.dropout is not None:
        	return self.pos_dropout(x) # [batch_size, seq_len, embedding_dim]
		else:
			return x
