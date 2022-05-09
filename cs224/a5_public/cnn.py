#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import max_pool1d

class CNN(nn.Module):

    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(self, char_embedding_size, num_filters, kernel_size=5, padding=1):
        """params:
          char_embedding_size (int): characters' embedding dimension
          num_filters (int): number of conv1d filters, also called number of output features, or output channels
          kernel_size (int): convolution window size
        """
        super(CNN, self).__init__()
        self.char_embedding_size = char_embedding_size
        self.conv = nn.Conv1d(in_channels=char_embedding_size, out_channels=num_filters,
                              kernel_size=kernel_size, padding=padding)

    def forward(self, x_reshaped):
        # map x_reshaped to x_conv_out
        assert x_reshaped.size()[-2] == self.char_embedding_size, "input tensor shape invalid, should be (n_words, char_embed, n_chars)"
        x_conv = self.conv(x_reshaped)
        x_conv = F.relu(x_conv)
        # MaxPool take the maximum across the second dimension
        x_conv_out = max_pool1d(x_conv, kernel_size=x_conv.size(2)).squeeze()
        return x_conv_out


    ### END YOUR CODE
