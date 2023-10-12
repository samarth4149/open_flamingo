import torch
import torch.nn as nn


class PromptLearner(nn.Module):
    def __init__(self, num_of_words, word_embedding, dtype):
        super().__init__()

        ctx_vectors = torch.empty(1, num_of_words, word_embedding, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

    def forward(self):
        return self.ctx