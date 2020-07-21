#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
    """
    Init the Highway module
    @param word_embed_size (int): Embedding size (dimensionality) for both the input (conv_output) and output
    """
    def __init__(self, word_embed_size, dropout_rate=0.5):
        super(Highway, self).__init__()
        self.W_project = nn.Linear(word_embed_size, word_embed_size)
        self.W_gate = nn.Linear(word_embed_size, word_embed_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x_conv_out: torch.tensor):
        # batched operation, map Xconv_out to Xhighway
        x_project = F.relu(self.W_project(x_conv_out))
        x_gate = torch.sigmoid(self.W_gate(x_conv_out))
        x_highway = x_gate * x_project + (1 -  x_gate) * x_conv_out
        return self.dropout(x_highway)



    ### END YOUR CODE
