import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):

    def __init__(self, d_model, d_k):
        super().__init__()

        self.WQ = nn.Linear(d_model, d_k)
        self.WK = nn.Linear(d_model, d_k)
        self.WV = nn.Linear(d_model, d_k)

    def forward(self, X):

        # -------------------------
        # 1. Create Q, K, V
        # -------------------------

        Q = self.WQ(X)
        K = self.WK(X)
        V = self.WV(X)

        # -------------------------
        # 2. Attention scores
        # -------------------------

        scores = Q @ K.T

        # -------------------------
        # 3. Scale
        # -------------------------

        scores = scores / torch.sqrt(
            torch.tensor(
                K.shape[-1],
                dtype=torch.float32
            )
        )

        # -------------------------
        # 4. Causal mask
        # -------------------------

        seq_len = X.shape[0]

        mask = torch.triu(
            torch.ones(seq_len, seq_len),
            diagonal=1
        )

        scores = scores.masked_fill(
            mask == 1,
            float("-inf")
        )

        # -------------------------
        # 5. Softmax
        # -------------------------

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # -------------------------
        # 6. Weighted values
        # -------------------------

        output = attention_weights @ V

        return output
X = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],   # cat
    [0.0, 1.0, 0.0, 1.0],   # eats
    [1.0, 1.0, 0.0, 0.0]    # food
])

attention = CausalSelfAttention(
    d_model=4,
    d_k=2
)

output = attention(X)

print("Output:")
print(output)

print("Shape:")
print(output.shape)    