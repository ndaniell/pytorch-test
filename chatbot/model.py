import torch
import torch.nn as nn

"""
Key Terms:
- RNN (Recurrent Neural Network): Neural network that processes sequences by maintaining a hidden state
- GRU (Gated Recurrent Unit): Advanced RNN variant that better handles long sequences
- Embedding: Converting discrete tokens (words) into continuous vector representations
- Hidden State: Internal memory that captures sequence information
- Encoder: Converts input sequence into a compressed representation (hidden state)
- Decoder: Generates output sequence from the encoder's hidden state
- LogSoftmax: Normalized log probabilities for model outputs
- Batch Size: Number of sequences processed together (1 in this implementation)
- Sequence Length: Length of input/output sequences (1 step at a time here)
- ReLU: Rectified Linear Unit activation function that helps neural networks learn non-linear patterns
"""

# Encoder network that takes input sequences and encodes them into a hidden state
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # Embedding layer converts input tokens to dense vectors
        self.embedding = nn.Embedding(input_size, hidden_size)
        # GRU layer processes the sequence and produces hidden states
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        # Convert input to embedding and reshape to (seq_len=1, batch=1, hidden_size)
        embedded = self.embedding(input).view(1, 1, -1)
        # Pass through GRU layer
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        # Initialize hidden state with zeros
        return torch.zeros(1, 1, self.hidden_size)

# Decoder network that generates output sequences from encoded state
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # Embedding layer for output tokens
        self.embedding = nn.Embedding(output_size, hidden_size)
        # GRU layer processes the sequence
        self.gru = nn.GRU(hidden_size, hidden_size)
        # Linear layer maps GRU output to vocabulary size
        self.out = nn.Linear(hidden_size, output_size)
        # LogSoftmax for output probabilities
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # Convert input to embedding and reshape
        output = self.embedding(input).view(1, 1, -1)
        # Apply ReLU activation
        output = torch.relu(output)
        # Pass through GRU
        output, hidden = self.gru(output, hidden)
        # Map to vocabulary size and apply softmax
        output = self.softmax(self.out(output[0]))
        return output, hidden