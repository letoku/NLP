from typing import Optional

import numpy as np
import torch
from abc import ABC, abstractmethod


class Layer(ABC):
    @abstractmethod
    def __call__(self, x:torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def parameters(self) -> list[torch.Tensor]:
        pass


class Linear(Layer):
    def __init__(self, in_features: int, out_features: int, g: torch.Generator=None, bias: bool=True):
        self.in_features = in_features
        self.out_features = out_features
        self.w = torch.empty(in_features, out_features)
        self.b: Optional[torch.Tensor] = None
        self.bias = bias
        self.g = g
        self._init_parameters()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.b is not None:
            return torch.matmul(x, self.w) + self.b
        else:
            return torch.matmul(x, self.w)

    def parameters(self) -> list[torch.Tensor]:
        if self.b is not None:
            return [self.w, self.b]
        else:
            return [self.w]

    def _init_parameters(self) -> None:
        k = 1 / self.in_features
        if self.g is not None:
            torch.nn.init.uniform_(self.w, -np.sqrt(k), np.sqrt(k), generator=self.g)
        else:
            torch.nn.init.uniform_(self.w, -np.sqrt(k), np.sqrt(k), generator=None)
        if self.bias:
            self.b = torch.zeros(self.out_features)


class ReLU(Layer):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)

    def parameters(self) -> list[torch.Tensor]:
        return []


class Tanh(Layer):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    def parameters(self) -> list[torch.Tensor]:
        return []


class Embedding(Layer):
    def __init__(self, num_embeddings: int, embedding_dim: int, g: torch.Generator=None):
        self.embedding_matrix = torch.randn(num_embeddings, embedding_dim, generator=g)
        self.g = g

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding_matrix[x]
        return emb.view(x.shape[0], -1)

    def parameters(self) -> list[torch.Tensor]:
        return [self.embedding_matrix]
