#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(self, char_embed_size, word_embed_size, max_word_len, kernel_size=5, padding=1):
        """ Init CNN module
        @param char_embed_size (int): The size of the character embeddings (dimensionality)
        @param word_embed_size (int): The size of the final word embedding (dimensionality)
        @param max_word_len (int): The length of longest word in the batch
        @param kernel_size (int): The kernel size for 1D convolutions
        @param padding (int): The padding to be applied to the input

        """
        super(CNN, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.in_channels = char_embed_size
        self.out_channels = word_embed_size
        self.max_pool_kernel = max_word_len-self.kernel_size+1
        self.conv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=True)
        self.maxpool = nn.MaxPool1d(kernel_size=self.max_pool_kernel)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Take a mini-batch of reshaped inputs(x_reshaped) and compute the output from the Conv1d network

        @param input (torch.Tensor): reshaped input to cnn with shape (max_sentence_length, batch_size, char_embed_size, max_word_len)
        @return (torch.Tensor): output tensor with shape (max_sentence_length, batch_size, word_embed_size) after passing
                                through the 1D convolution network
        """
        x_conv = self.conv(input)
        x_conv_out = self.maxpool(F.relu(x_conv))
        return x_conv_out

    ### END YOUR CODE

