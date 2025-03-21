from typing import Optional, Tuple, List, Any

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
    def parameters(self) -> List[torch.Tensor]:
        pass

    def n_parameters(self) -> int:
        params = self.parameters()
        n_params = 0
        for param in params:
            n_params += param.numel()
        return n_params

    @abstractmethod
    def __call__(self, *args) -> Tuple[Any, ...]:
        pass


class ForwardLayer(Layer, ABC):
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass

class StateDependentLayer(Layer, ABC):
    @abstractmethod
    def __call__(self, x: torch.Tensor, hidden_state: torch) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class ReLU(ForwardLayer):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)

    def parameters(self) -> List[torch.Tensor]:
        return []


class Tanh(ForwardLayer):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    def parameters(self) -> List[torch.Tensor]:
        return []


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

    def parameters(self) -> List[torch.Tensor]:
        params = [self.w]
        if self.b is not None:
            params += [self.b]
        return params

    def _init_parameters(self) -> None:
        _init_weights(self.w, self.in_features, self.g)
        if self.bias:
            self.b = torch.zeros(self.out_features)


class RecurrentLayer(StateDependentLayer):
    ALLOWED_NON_LINEARITIES = {
        "ReLU": ReLU,
        "Tanh": Tanh,
    }

    def __init__(self, in_features: int, hidden_state_features: int, out_features: int,
                 different_weights_for_hidden_output: bool = False,
                 non_linearity_type: str = "ReLU",
                 g: torch.Generator=None,
                 bias: bool=True):
        self.in_features = in_features
        self.hidden_state_features = hidden_state_features
        self.out_features = out_features
        self.w_in = torch.empty(in_features, out_features)
        self.w_hidden = torch.empty(hidden_state_features, hidden_state_features)
        self.b: Optional[torch.Tensor] = None

        self.w_in_for_hidden_pass: Optional[torch.Tensor] = None
        self.w_hidden_for_hidden_pass: Optional[torch.Tensor] = None
        self.b_for_hidden_pass: Optional[torch.Tensor] = None

        self.bias = bias
        self.different_weights_for_hidden_output = different_weights_for_hidden_output
        self.g = g
        self._init_parameters()
        self._set_non_linearity(non_linearity_type)


    def __call__(self, x: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = torch.matmul(x, self.w_in) + torch.matmul(hidden, self.w_hidden)
        if self.b is not None:
            out += self.b

        if self.different_weights_for_hidden_output:
            hidden = torch.matmul(x, self.w_in_for_hidden_pass) + torch.matmul(hidden, self.w_hidden_for_hidden_pass)
            if self.b_for_hidden_pass is not None:
                hidden = self.non_linearity(hidden)
        else:
            hidden = self.non_linearity(out)

        return out, hidden

    def parameters(self) -> List[torch.Tensor]:
        params = [self.w_in, self.w_hidden]
        if self.b is not None:
            params += [self.b]
        if self.different_weights_for_hidden_output:
            params += [self.w_hidden_for_hidden_pass, self.w_in_for_hidden_pass, self.b_for_hidden_pass]

        return params

    def _init_parameters(self) -> None:
        _init_weights(self.w_in, self.in_features, self.g)
        _init_weights(self.w_hidden, self.hidden_state_features, self.g)  # TODO: Experiment with this later on.
        if self.different_weights_for_hidden_output:
            self.w_in_for_hidden_pass = torch.empty(self.in_features, self.out_features)
            self.w_hidden_for_hidden_pass = torch.empty(self.hidden_state_features, self.hidden_state_features)

            _init_weights(self.w_hidden_for_hidden_pass, self.hidden_state_features, self.g)
            _init_weights(self.w_in_for_hidden_pass, self.in_features, self.g)
            if self.bias:
                self.b_for_hidden_pass = torch.zeros(self.hidden_state_features)

        if self.bias:
            self.b = torch.zeros(self.out_features)

    def _set_non_linearity(self, non_linearity_type: str):
        if non_linearity_type not in RecurrentLayer.ALLOWED_NON_LINEARITIES:
            raise KeyError(
                f"Nonlinearity type '{non_linearity_type}' is not in ALLOWED_NON_LINEARITIES. Available nonlinearities: {list(RecurrentLayer.ALLOWED_NON_LINEARITIES.keys())}")

        self.non_linearity = RecurrentLayer.ALLOWED_NON_LINEARITIES[non_linearity_type]()


class Embedding(ForwardLayer):
    def __init__(self, num_embeddings: int, embedding_dim: int, g: torch.Generator=None):
        self.embedding_matrix = torch.randn(num_embeddings, embedding_dim, generator=g)
        self.g = g

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding_matrix[x]
        return emb.view(x.shape[0], -1)

    def parameters(self) -> List[torch.Tensor]:
        return [self.embedding_matrix]
