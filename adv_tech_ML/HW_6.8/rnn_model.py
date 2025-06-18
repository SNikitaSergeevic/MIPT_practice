import torch
import torch.nn as nn

embedding_dim = 5
hidden_dim = 9

class RNNWithAttentionModel(nn.Module):
   def __init__(self, random_seed=5):
       super(RNNWithAttentionModel, self).__init__()
       torch.manual_seed(random_seed)
       torch.cuda.manual_seed(random_seed)
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False
       # Create an embedding layer for the vocabulary
       # your code here
       # Create an RNN layer
       # your code here
       # Apply a linear transformation to get the attention scores
       # your code here
       self.fc = nn.Linear(hidden_dim, vocab_size)
   def forward(self, x):
       x = self.embeddings(x)
       out, _ = self.rnn(x)
       attention_out = self.attention(out).squeeze(2)
       #  Get the attention weights
       # your code here
       # Compute the context vector
       # your code here
       out = self.fc(context)
       return out
