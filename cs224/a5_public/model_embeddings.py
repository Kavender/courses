#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ### YOUR CODE HERE for part 1h
        self.word_embed_size = word_embed_size
        self.char_embed_size = 50
        self.vocab = vocab
        self.dropout = nn.Dropout(0.3)
        self.X_char_embd = nn.Embedding(num_embeddings = len(self.vocab.char2id), embedding_dim=self.char_embed_size,
                                        padding_idx=self.vocab.char_pad)
        self.CNN_model = CNN(char_embedding_size = self.char_embed_size, num_filters=word_embed_size)
        self.highway = Highway(word_embed_size=word_embed_size)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h
        x = self.X_char_embd(input)

        sent_len, batch_size, max_word, char_embed_size = x.shape
        view_shape = (sent_len*batch_size, max_word, char_embed_size)
        x_reshaped = x.view(view_shape).transpose(1, 2)

        x_conv_out = self.CNN_model(x_reshaped)
        x_highway = self.highway(x_conv_out)
        x_embedding = self.dropout(x_highway)

        return x_embedding.view(sent_len, batch_size, self.word_embed_size)
        ### END YOUR CODE