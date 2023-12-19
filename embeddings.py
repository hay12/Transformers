"""
embeddings.py

Introduction:
--------------
This file is responsible for all aspects related to embeddings within the Transformer architecture,
as outlined in the "Attention Is All You Need" paper. It serves to manage the following components:

1. Token Embeddings: 
    - Converts tokens (words, sub-words, or characters) into high-dimensional vectors. These vectors serve as the initial 
      representation of the tokens and are subsequently used by the Transformer's encoder and decoder.

2. Positional Encodings:
    - Generates positional encodings to capture the order of tokens in a sequence, a feature not inherently captured by 
      standard embeddings. These are added to the token embeddings to form the final input embeddings for the model.

By centralizing the management of these critical components, this file ensures consistent and reusable embedding 
functionality across the Transformer's encoder and decoder modules.

Classes and Functions:
-----------------------
1. TokenEmbedding: A class for generating token embeddings.
2. PositionalEncoding: A class for creating positional encodings.
3. combine_embeddings: A function to combine token and positional embeddings.

Author: Hay Hoffman
Date: 31/8/23
"""

# Importing necessary libraries
import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    """
    TokenEmbedding Class
    
    This class is responsible for converting token IDs into high-dimensional vectors (embeddings).
    The embeddings serve as the initial representation of tokens and are used by both the encoder and decoder
    in the Transformer architecture.
    
    Attributes:
    -----------
    vocab_size : int
        The size of the vocabulary, i.e., the number of unique tokens.
        
    d_model : int
        The dimension of the embedding vectors.
        
    Methods:
    --------
    forward(x):
        Takes a tensor of token IDs (x) and returns their corresponding embeddings.
    """
    
    def __init__(self, vocab_size: int = 5000, d_model: int = 512):
        """
        Constructor for the TokenEmbedding class.
        
        Parameters:
        -----------
        vocab_size : int
            The size of the vocabulary, i.e., the number of unique tokens.
            
        d_model : int
            The dimension of the embedding vectors.
        """
        super(TokenEmbedding, self).__init__()
        
        self.d_model = d_model
        self.embedding_scale_factor = math.sqrt(self.d_model)
        
        # Define the embedding layer
        self.embedding = nn.Embedding(vocab_size, self.d_model)
        
    def forward(self, x: torch.Tensor)->torch.Tensor:
        """
        Forward pass for converting token IDs to embeddings.
        
        Parameters:
        -----------
        x : torch.Tensor
            A tensor containing token IDs. Shape: [batch_size, sequence_length]
            
        Returns:
        --------
        torch.Tensor
            A tensor containing the embeddings corresponding to the input token IDs.
            Shape: [batch_size, sequence_length, d_model]
        """
        return self.embedding(x) * self.embedding_scale_factor
    
class PositionalEncoding(nn.Module):
    """
    PositionalEncoding Class
    
    This class is responsible for generating positional encodings to capture the order of tokens in a sequence.
    These encodings are added to the token embeddings to form the final input embeddings for the model.
    
    Attributes:
    -----------
    d_model : int
        The dimension of the embeddings (also the dimension of the positional encodings).
        
    init_len : int
        The maximum sequence length for which positional encodings are precomputed.
        
    pos_encodings : torch.Tensor
        Precomputed positional encodings up to init_len.
        
    Methods:
    --------
    forward(x):
        Takes a tensor of embeddings (x) and returns the embeddings with added positional encodings.
        
    generate_positional_encodings(seq_len, d_model):
        Generates positional encodings for a given sequence length and model dimension.
    """
    
    def __init__(self, d_model:int = 512, init_len:int = 5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.init_len = init_len
        self.pos_encodings = self.generate_positional_encodings(self.init_len, self.d_model)
        
        # Register the positional encodings as a buffer
        self.register_buffer('pos_encodings', self.pos_encodings)

    def generate_positional_encodings(self, seq_len:int, d_model:int) -> torch.Tensor:
        """
        Generates positional encodings for a given sequence length and model dimension.
        
        Parameters:
        -----------
        seq_len : int
            The sequence length for which to generate positional encodings.
            
        d_model : int
            The model dimension.
            
        Returns:
        --------
        torch.Tensor
            A tensor containing the generated positional encodings.
        """
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_encodings = torch.zeros(seq_len, d_model)
        pos_encodings[:, 0::2] = torch.sin(position * div_term)
        pos_encodings[:, 1::2] = torch.cos(position * div_term)
        
        pos_encodings = pos_encodings.unsqueeze(0)
        
        return pos_encodings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for adding positional encodings to input embeddings.
        
        Parameters:
        -----------
        x : torch.Tensor
            The input embeddings of shape [batch_size, sequence_length, d_model]
            
        Returns:
        --------
        torch.Tensor
            The input embeddings with added positional encodings.
        """
        
        # Add positional encodings to each sequence in the batch
        # Here, broadcasting takes care of the batch_size dimension
        encoded_vector = x + self.pos_encodings
        
        return encoded_vector

if __name__ == "__main__":
    print("This file is intended to be imported as a module and not to be run directly.")
    
