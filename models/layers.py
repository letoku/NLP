from typing import Optional, Tuple, Any

import numpy as np
import torch
from abc import ABC, abstractmethod


def _init_weights(w: torch.Tensor, in_features: int, g: torch.Generator=None) -> None:
    k = 1 / in_features
    if g is not None:
        torch.nn.init.uniform_(w, -np.sqrt(k), np.sqrt(k), generator=g)
    else:
        torch.nn.init.uniform_(w, -np.sqrt(k), np.sqrt(k), generator=None)


class Layer(ABC):
    @abstractmethod
    def parameters(self) -> list[torch.Tensor]:
        pass

    @abstractmethod
    def __call__(self, *args) -> Tuple[Any, ...]:
        pass


class ForwardLayer(Layer, ABC):
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass

class StateDependentLayer(Layer, ABC):
    @abstractmethod
    def __call__(self, x: torch.Tensor, hidden_state: torch) -> torch.Tensor:
        pass


class Linear(ForwardLayer):
    def __init__(self, in_features: int, out_features: int, g: torch.Generator=None, bias: bool=True):
        self.in_features = in_features
        self.out_features = out_features
        self.w = torch.empty(in_features, out_features)
        self.b: Optional[torch.Tensor] = None
        self.bias = bias
        self.g = g
        self._init_parameters()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(x, self.w)
        if self.b is not None:
            out += self.b
        return out

    def parameters(self) -> list[torch.Tensor]:
        if self.b is not None:
            return [self.w, self.b]
        else:
            return [self.w]

    def _init_parameters(self) -> None:
        _init_weights(self.w, self.in_features, self.g)
        if self.bias:
            self.b = torch.zeros(self.out_features)


class RecurrentLayer(StateDependentLayer):
    def __init__(self, in_features: int, hidden_state_features: int, out_features: int, g: torch.Generator=None,
                 bias: bool=True):
        self.in_features = in_features
        self.hidden_state_features = hidden_state_features
        self.out_features = out_features
        self.w_in = torch.empty(in_features, out_features)
        self.w_hidden = torch.empty(hidden_state_features, out_features)
        self.b: Optional[torch.Tensor] = None
        self.bias = bias
        self.g = g
        self._init_parameters()

    def __call__(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(x, self.w_in) + torch.matmul(hidden, self.w_hidden)
        if self.b is not None:
            out += self.b
        return out

    def parameters(self) -> list[torch.Tensor]:
        if self.b is not None:
            return [self.w_in, self.w_hidden, self.b]
        else:
            return [self.w_in, self.w_hidden]

    def _init_parameters(self) -> None:
        _init_weights(self.w_in, self.in_features, self.g)
        _init_weights(self.w_hidden, self.hidden_state_features, self.g)  # TODO: Experiment with this later on.
        if self.bias:
            self.b = torch.zeros(self.out_features)


class ReLU(ForwardLayer):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)

    def parameters(self) -> list[torch.Tensor]:
        return []


class Tanh(ForwardLayer):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    def parameters(self) -> list[torch.Tensor]:
        return []


class Embedding(ForwardLayer):
    def __init__(self, num_embeddings: int, embedding_dim: int, g: torch.Generator=None):
        self.embedding_matrix = torch.randn(num_embeddings, embedding_dim, generator=g)
        self.g = g

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding_matrix[x]
        return emb.view(x.shape[0], -1)

    def parameters(self) -> list[torch.Tensor]:
        return [self.embedding_matrix]
