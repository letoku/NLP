from abc import ABC, abstractmethod
from typing import Tuple
import torch.nn.functional as F
from .layers import *


ALLOWED_LAYERS = {
    "Linear": Linear,
    "ReLU": ReLU,
    "Tanh": Tanh,
    "Embedding": Embedding,
    "Recurrent": RecurrentLayer
}


class Module(ABC):
    def __init__(self, specs: list[tuple[str, dict]], g: torch.Generator=None, last_layer_scaling: float = 0.01):
        self.parameters: list[torch.Tensor] = []
        self.layers: list[Layer] = []
        self.g = g

        for spec in specs:
            layer_type, kwargs = spec
            if layer_type not in ALLOWED_LAYERS:
                raise KeyError(f"Layer type '{layer_type}' is not in ALLOWED_LAYERS. Available layers: {list(ALLOWED_LAYERS.keys())}")

            added_layer = ALLOWED_LAYERS[layer_type]
            self.layers.append(added_layer(**kwargs))

        self._scale_layer(-1, last_layer_scaling)
        for layer in self.layers:
            self.parameters += layer.parameters()

    @abstractmethod
    def __call__(self, *args) -> Tuple[Any, ...]:
        pass

    @abstractmethod
    def predict_proba(self, *args) -> Tuple[Any, ...]:
        pass

    def _scale_layer(self, layer_index: int, scaling_factor: float):
        for param in self.layers[layer_index].parameters():
            param *= scaling_factor

    def set_train_mode(self) -> None:
        for param in self.parameters:
            param.requires_grad = True

    def set_eval_mode(self) -> None:
        for param in self.parameters:
            param.requires_grad = False


class ForwardModule(Module):
    def __init__(self, specs: list[tuple[str, dict]], g: torch.Generator = None):
        super().__init__(specs, g)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        out = self(x)
        return F.softmax(out, dim=-1)


class StateDependentModule(Module):
    def __init__(self, specs: list[tuple[str, dict]], hidden_dim: int, g: torch.Generator=None):
        super().__init__(specs, g)
        self.hidden_dims = (hidden_dim, -1)
        self._set_hidden_dims()

    def __call__(self, inputs: torch.Tensor, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = inputs
        hidden_layer_counter = 0
        for i, layer in enumerate(self.layers):
            if issubclass(layer.__class__, StateDependentLayer):
                h_t = hidden_states[:, :, hidden_layer_counter].clone()  # Prevent in-place modification.
                x, hidden_states[:, :, hidden_layer_counter] = layer(x, h_t)
                hidden_layer_counter += 1
            else:
                x = layer(x)
        return x, hidden_states

    def predict_proba(self, inputs: torch.Tensor, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, hidden_states = self(inputs, hidden_states)
        return F.softmax(out, dim=-1), hidden_states

    def _set_hidden_dims(self) -> None:
        hidden_layers = 0
        for layer in self.layers:
            if issubclass(layer.__class__, StateDependentLayer):
                hidden_layers += 1

        self.hidden_dims = (self.hidden_dims[0], hidden_layers)

    def get_hidden_states_dims(self) -> Tuple[int, int]:
        return self.hidden_dims
