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
    def __init__(self, embed_size):
        """ Init Highway Layer
        @param embed_size (int): The size of the final word embedding (dimensionality)

        """
        super(Highway, self).__init__()
        self.proj = nn.Linear(in_features = embed_size, out_features = embed_size, bias = True)
        self.gate = nn.Linear(in_features = embed_size, out_features = embed_size, bias = True)

    def forward(self, conv_out: torch.Tensor) -> torch.Tensor:
        """ Take a mini-batch of CNN outputs and compute the output from the highway layer

        @param conv_out (torch.Tensor): output from the CNN with shape (max_sentence_length, batch_size, embed_size)
        @return (torch.Tensor): output tensor with shape (max_sentence_length, batch_size, embed_size) after passing
                                through the highway layer
        """
        x_proj = F.relu(self.proj(conv_out))
        x_gate = torch.sigmoid(self.gate(conv_out))
        x_highway = (x_gate*x_proj) + ((1-x_gate)*conv_out)
        return x_highway

    ### END YOUR CODE

