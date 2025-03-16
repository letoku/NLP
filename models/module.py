from abc import ABC, abstractmethod
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
    def __init__(self, specs: list[tuple[str, dict]], g: torch.Generator=None):
        self.parameters: list[torch.Tensor] = []
        self.layers: list[Layer] = []
        self.g = g

        for spec in specs:
            layer_type, kwargs = spec
            assert layer_type in ALLOWED_LAYERS
            added_layer = ALLOWED_LAYERS[layer_type]
            self.layers.append(added_layer(**kwargs))

        for layer in self.layers:
            self.parameters += layer.parameters()

    @abstractmethod
    def __call__(self, *args) -> Tuple[Any, ...]:
        pass

    @abstractmethod
    def predict_proba(self, *args) -> Tuple[Any, ...]:
        pass

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

    def set_train_mode(self) -> None:
        for param in self.parameters:
            param.requires_grad = True

    def set_eval_mode(self) -> None:
        for param in self.parameters:
            param.requires_grad = False


class StateDependentModule(Module):
    """
    Assume that after each recurrent layer there is nonlinear layer.
    """
    def __init__(self, specs: list[tuple[str, dict]], g: torch.Generator=None):
        super().__init__(specs, g)

    def __call__(self, inputs: torch.Tensor, hidden_states: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x = inputs
        hidden_layer_counter = 0
        for i, layer in enumerate(self.layers):
            if issubclass(layer.__class__, StateDependentLayer):
                h_t = hidden_states[:, :, hidden_layer_counter]
                x, hidden_states[:, :, hidden_layer_counter] = layer(x, h_t)
                hidden_states[:, :, hidden_layer_counter] = self.layers[i + 1](hidden_states[:, :, hidden_layer_counter])  # Here we assume that next is nonlinear layer.
                hidden_layer_counter += 1
            else:
                x = layer(x)
        return x, hidden_states

    def predict_proba(self, inputs: torch.Tensor, hidden_states: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        out, hidden_states = self(inputs, hidden_states)
        return F.softmax(out, dim=-1), hidden_states
