from transformers import AutoTokenizer
from transformers import AutoModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "I love learning artificial intelligence with unhappiness!"

tokens = tokenizer.tokenize(text)

print(tokens)
token_ids = tokenizer.encode(text)

print(token_ids)

vocab=tokenizer.get_vocab()
# print("length", len(vocab))
# print(vocab["learn"])


model = AutoModel.from_pretrained("bert-base-uncased")

embedding_layer = model.get_input_embeddings()

import torch

token_id = 2293

with torch.no_grad():
    vector = embedding_layer(torch.tensor([token_id]))

print(vector.shape)
text = "I love learning artificial intelligence!"

inputs = tokenizer(text, return_tensors="pt")

print(inputs["input_ids"])
with torch.no_grad():
    embeddings = embedding_layer(inputs["input_ids"])

print("emb shape",embeddings.shape)