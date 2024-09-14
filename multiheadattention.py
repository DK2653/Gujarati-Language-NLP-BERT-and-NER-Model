import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        # Check if embed_size is divisible by num_heads
        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads"

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # Linear transformation
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1,
                                                                          3)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))

        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(attention, V)

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.embed_size)

        # Final linear transformation
        x = self.fc_out(x)

        return x

# Define the parameters for MultiHeadAttention
embed_size = 128
num_heads = 8
seq_len = 20
batch_size = 4

# Create dummy input tensors
query = torch.randn(batch_size, seq_len, embed_size)  # (batch_size, seq_len, embed_size)
key = torch.randn(batch_size, seq_len, embed_size)    # (batch_size, seq_len, embed_size)
value = torch.randn(batch_size, seq_len, embed_size)  # (batch_size, seq_len, embed_size)

# Instantiate MultiHeadAttention module
multihead_attn = MultiHeadAttention(embed_size, num_heads)

# Perform forward pass
output = multihead_attn(query, key, value)

# Print output shape
print("Output shape:", output.shape)  # Expected: (batch_size, seq_len, embed_size)

# Save the model
torch.save(multihead_attn.state_dict(), 'multihead_attention.pth')

print("Model saved successfully!")
