import numpy as np


# ============================================================
# 1. INPUT
# ============================================================

# 3 tokens:
# cat, eats, food
#
# Each token has 4 features.
#
# Shape: (3, 4)

X = np.array([
    [1.0, 0.0, 1.0, 0.0],   # cat
    [0.0, 1.0, 0.0, 1.0],   # eats
    [1.0, 1.0, 0.0, 0.0]    # food
])

print("Input X shape:", X.shape)


# ============================================================
# 2. SOFTMAX FUNCTION
# ============================================================

# Converts attention scores into attention weights.
# Each row will add up to approximately 1.

def softmax(x):
    exp_x = np.exp(
        x - np.max(x, axis=1, keepdims=True)
    )

    return exp_x / np.sum(
        exp_x,
        axis=1,
        keepdims=True
    )


# ============================================================
# 3. HEAD 1 - Q, K, V WEIGHTS
# ============================================================

WQ1 = np.array([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6],
    [0.7, 0.8]
])

WK1 = np.array([
    [0.2, 0.1],
    [0.4, 0.3],
    [0.6, 0.5],
    [0.8, 0.7]
])

WV1 = np.array([
    [0.1, 0.3],
    [0.2, 0.4],
    [0.5, 0.7],
    [0.6, 0.8]
])


# Create Query, Key and Value for Head 1

Q1 = X @ WQ1
K1 = X @ WK1
V1 = X @ WV1


# ============================================================
# 4. HEAD 2 - Q, K, V WEIGHTS
# ============================================================

WQ2 = np.array([
    [0.8, 0.7],
    [0.6, 0.5],
    [0.4, 0.3],
    [0.2, 0.1]
])

WK2 = np.array([
    [0.7, 0.8],
    [0.5, 0.6],
    [0.3, 0.4],
    [0.1, 0.2]
])

WV2 = np.array([
    [0.6, 0.8],
    [0.5, 0.7],
    [0.2, 0.4],
    [0.1, 0.3]
])


# Create Query, Key and Value for Head 2

Q2 = X @ WQ2
K2 = X @ WK2
V2 = X @ WV2


# ============================================================
# 5. ATTENTION - HEAD 1
# ============================================================

# Step 1: Compare Query with every Key

score1 = Q1 @ K1.T


# Step 2: Scale the scores
# sqrt(d_k) = sqrt(2)

score1 = score1 / np.sqrt(2)


# Step 3: Convert scores into attention weights

att1 = softmax(score1)


# Step 4: Use attention weights to combine Values

head1 = att1 @ V1


# ============================================================
# 6. ATTENTION - HEAD 2
# ============================================================

# Same attention process for Head 2

score2 = Q2 @ K2.T

score2 = score2 / np.sqrt(2)

att2 = softmax(score2)

head2 = att2 @ V2


# ============================================================
# 7. MULTI-HEAD ATTENTION
# ============================================================

print("\n--- Multi-Head Attention ---")

print("Q1:", Q1.shape)
print("K1:", K1.shape)
print("V1:", V1.shape)

print("Q2:", Q2.shape)
print("K2:", K2.shape)
print("V2:", V2.shape)

print("Head 1:", head1.shape)
print("Head 2:", head2.shape)


# Combine the outputs of both heads

combined = np.concatenate(
    [head1, head2],
    axis=1
)

print("\nHead 1:")
print(head1)

print("\nHead 2:")
print(head2)

print("\nCombined:")
print(combined)

print("Combined shape:", combined.shape)


# ============================================================
# 8. OUTPUT PROJECTION (WO)
# ============================================================

# WO mixes information from the different attention heads.

WO = np.array([
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.2, 0.3, 0.4, 0.5],
    [0.6, 0.7, 0.8, 0.9]
])


# Final Multi-Head Attention output

output = combined @ WO

print("\nAttention Output:")
print(output)
print("Shape:", output.shape)


# ============================================================
# 9. FIRST RESIDUAL CONNECTION
# ============================================================

# Add the original input to the attention output.
#
# Purpose:
# Keep the previous information while adding
# the new information learned through attention.

residual = X + output

print("\nResidual:")
print(residual)
print("Shape:", residual.shape)


# ============================================================
# 10. LAYER NORMALIZATION
# ============================================================

def layernorm(x):

    # Calculate mean for each token
    mean = np.mean(
        x,
        axis=1,
        keepdims=True
    )

    # Calculate variance for each token
    variance = np.var(
        x,
        axis=1,
        keepdims=True
    )

    # Normalize
    normalized = (
        (x - mean)
        / np.sqrt(variance + 1e-5)
    )

    return normalized


# Apply LayerNorm

normalised = layernorm(residual)

print("\nLayerNorm Output:")
print(normalised)
print("Shape:", normalised.shape)


# ============================================================
# 11. FEED-FORWARD NETWORK (FFN)
# ============================================================

# First FFN layer:
#
# 4 dimensions → 8 dimensions

W1 = np.array([
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
])

b1 = np.zeros(8)


# Second FFN layer:
#
# 8 dimensions → 4 dimensions

W2 = np.array([
    [0.1, 0.2, 0.3, 0.4],
    [0.2, 0.3, 0.4, 0.5],
    [0.3, 0.4, 0.5, 0.6],
    [0.4, 0.5, 0.6, 0.7],
    [0.5, 0.6, 0.7, 0.8],
    [0.6, 0.7, 0.8, 0.9],
    [0.7, 0.8, 0.9, 1.0],
    [0.8, 0.9, 1.0, 1.1]
])

b2 = np.zeros(4)


# ============================================================
# 12. FFN - FIRST LINEAR LAYER
# ============================================================

ffn_hidden = normalised @ W1 + b1

print("\nFFN Hidden:")
print(ffn_hidden)
print("Shape:", ffn_hidden.shape)


# ============================================================
# 13. RELU
# ============================================================

# Remove negative values

ffn_hidden = np.maximum(
    0,
    ffn_hidden
)

print("\nAfter ReLU:")
print(ffn_hidden)


# ============================================================
# 14. FFN - SECOND LINEAR LAYER
# ============================================================

ffn_output = ffn_hidden @ W2 + b2

print("\nFFN Output:")
print(ffn_output)
print("Shape:", ffn_output.shape)


# ============================================================
# 15. SECOND RESIDUAL CONNECTION
# ============================================================

# Keep the representation from before FFN
# and add the new information produced by FFN.

residual2 = normalised + ffn_output

print("\nResidual 2:")
print(residual2)
print("Shape:", residual2.shape)


# ============================================================
# 16. FINAL LAYER NORMALIZATION
# ============================================================

normalised2 = layernorm(residual2)

print("\nFinal Transformer Block Output:")
print(normalised2)
print("Shape:", normalised2.shape)


# ============================================================
# COMPLETE TRANSFORMER ENCODER BLOCK
# ============================================================

# X
# ↓
# Multi-Head Attention
# ↓
# Add X (Residual)
# ↓
# LayerNorm
# ↓
# FFN
# ↓
# Add previous representation (Residual)
# ↓
# LayerNorm
# ↓
# Final Transformer Block Output