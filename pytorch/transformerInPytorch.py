import torch
import torch.nn as nn

class Multihead(nn.module):
    def __init__(self,d_model,nums_head):
        