from typing import Self
import torch.nn.functional as F
from .layers import *


ALLOWED_LAYERS = {
    "Linear": Linear,
    "ReLU": ReLU,
    "Tanh": Tanh,
    "Embedding": Embedding,
    "Recurrent": RecurrentLayer,
    "LSTM": LSTMLayer,
    "GRU": GRULayer
}


class Module(ABC):
    def __init__(self, specs: list[tuple[str, dict]], g: torch.Generator=None, last_layer_scaling: float=0.01):
        self.layers: list[Layer] = []
        self.g = g

        for spec in specs:
            layer_type, kwargs = spec
            if layer_type not in ALLOWED_LAYERS:
                raise KeyError(f"Layer type '{layer_type}' is not in ALLOWED_LAYERS. Available layers: {list(ALLOWED_LAYERS.keys())}")

            added_layer = ALLOWED_LAYERS[layer_type]
            self.layers.append(added_layer(**kwargs))

        self._scale_layer(-1, last_layer_scaling)

    @abstractmethod
    def __call__(self, *args) -> Tuple[Any, ...]:
        pass

    @abstractmethod
    def predict_proba(self, *args) -> Tuple[Any, ...]:
        pass

    @property
    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params

    def n_parameters(self) -> int:
        n_params = 0
        for layer in self.layers:
            n_params += layer.n_parameters()
        return n_params

    def to(self, device: torch.device) -> Self:
        for layer in self.layers:
            layer.to(device)
        return self

    def _scale_layer(self, layer_index: int, scaling_factor: float):
        for param in self.layers[layer_index].parameters():
            param *= scaling_factor

    def set_train_mode(self) -> None:
        for layer in self.layers:
            layer.set_train_mode()

    def set_eval_mode(self) -> None:
        for layer in self.layers:
            layer.set_eval_mode()


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


class ContextAndStateDependentModule(Module):
    def __init__(self, specs: list[tuple[str, dict]], hidden_dim: int, g: torch.Generator=None):
        super().__init__(specs, g)
        self.hidden_dims = (hidden_dim, -1)
        self._set_hidden_dims()

    def __call__(self, inputs: torch.Tensor, hidden_states: torch.Tensor, context: torch.Tensor) ->\
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = inputs
        hidden_layer_counter = 0
        for i, layer in enumerate(self.layers):
            if issubclass(layer.__class__, ContextAndStateDependentLayer):
                h_t = hidden_states[:, :, hidden_layer_counter].clone()  # Prevent in-place modification.
                c_t = context[:, :, hidden_layer_counter].clone()
                x, hidden_states[:, :, hidden_layer_counter], context[:, :, hidden_layer_counter] = layer(x, h_t, c_t)
                hidden_layer_counter += 1
            else:
                x = layer(x)
        return x, hidden_states, context

    def predict_proba(self, inputs: torch.Tensor, hidden_states: torch.Tensor, context: torch.Tensor) ->\
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out, hidden_states, context = self(inputs, hidden_states, context)
        return F.softmax(out, dim=-1), hidden_states, context

    def _set_hidden_dims(self) -> None:
        hidden_layers = 0
        for layer in self.layers:
            if issubclass(layer.__class__, ContextAndStateDependentLayer):
                hidden_layers += 1

        self.hidden_dims = (self.hidden_dims[0], hidden_layers)

    def get_hidden_states_dims(self) -> Tuple[int, int]:
        return self.hidden_dims