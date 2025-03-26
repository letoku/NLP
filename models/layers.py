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

    def to(self, device: torch.device) -> None:
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                setattr(self, attr_name, attr_value.to(device))

    def set_train_mode(self) -> None:
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                attr_value.requires_grad = True

    def set_eval_mode(self) -> None:
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                attr_value.requires_grad = False

    @abstractmethod
    def __call__(self, *args) -> Tuple[Any, ...]:
        pass


class ForwardLayer(Layer, ABC):
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass


class ContextAndStateDependentLayer(Layer, ABC):
    @abstractmethod
    def __call__(self, x: torch.Tensor, hidden_state: torch.Tensor, context: torch.Tensor) ->\
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


class StateDependentLayer(Layer, ABC):
    ALLOWED_NON_LINEARITIES = {
        "ReLU": ReLU,
        "Tanh": Tanh,
    }

    @abstractmethod
    def __call__(self, x: torch.Tensor, hidden_state: torch) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def _set_non_linearity(self, non_linearity_type: str):
        if non_linearity_type not in StateDependentLayer.ALLOWED_NON_LINEARITIES:
            raise KeyError(
                f"Nonlinearity type '{non_linearity_type}' is not in ALLOWED_NON_LINEARITIES. Available nonlinearities:"
                f" {list(StateDependentLayer.ALLOWED_NON_LINEARITIES.keys())}")

        self.non_linearity = StateDependentLayer.ALLOWED_NON_LINEARITIES[non_linearity_type]()



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


class Embedding(ForwardLayer):
    def __init__(self, num_embeddings: int, embedding_dim: int, g: torch.Generator=None):
        self.embedding_matrix = torch.randn(num_embeddings, embedding_dim, generator=g)
        self.g = g

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding_matrix[x]
        return emb.view(x.shape[0], -1)

    def parameters(self) -> List[torch.Tensor]:
        return [self.embedding_matrix]


class RecurrentLayer(StateDependentLayer):
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


class GRULayer(StateDependentLayer):
    def __init__(self, in_features: int, hidden_state_features: int,
                 non_linearity_type: str = "ReLU",
                 g: torch.Generator=None):
        self.in_features = in_features
        self.hidden_state_features = hidden_state_features

        # reset gate
        self.w_xr = torch.empty(in_features, hidden_state_features)
        self.w_hr = torch.empty(hidden_state_features, hidden_state_features)
        self.b_r = torch.zeros(hidden_state_features)

        # add gate
        self.w_xz = torch.empty(in_features, hidden_state_features)
        self.w_hz = torch.empty(hidden_state_features, hidden_state_features)
        self.b_z = torch.zeros(hidden_state_features)

        # updated hidden
        self.w_x_uh = torch.empty(in_features, hidden_state_features)
        self.w_rh_uh = torch.empty(hidden_state_features, hidden_state_features)
        self.b_uh = torch.zeros(hidden_state_features)

        self.g = g
        self._init_parameters()
        self._set_non_linearity(non_linearity_type)

    def __call__(self, x: torch.Tensor, hidden_state: torch) -> Tuple[torch.Tensor, torch.Tensor]:
        # reset gate
        r = torch.sigmoid(torch.matmul(x, self.w_xr) + torch.matmul(hidden_state, self.w_hr) + self.b_r)

        # h after gating with r
        rh = r * hidden_state

        # updated h
        uh = self.non_linearity(torch.matmul(rh, self.w_rh_uh) + torch.matmul(x, self.w_x_uh) + self.b_uh)

        # add gate
        z = torch.sigmoid(torch.matmul(x, self.w_xz) + torch.matmul(hidden_state, self.w_hz) + self.b_z)

        out = z * hidden_state + (1 - z) * uh

        return out, out

    def parameters(self) -> List[torch.Tensor]:
        return [self.w_xr, self.w_hr, self.b_r,
                self.w_xz, self.w_hz, self.b_z,
                self.w_x_uh, self.w_rh_uh, self.b_uh]

    def _init_parameters(self):
        _init_weights(self.w_xr, self.in_features, self.g)
        _init_weights(self.w_hr, self.hidden_state_features, self.g)

        _init_weights(self.w_xz, self.in_features, self.g)
        _init_weights(self.w_hz, self.hidden_state_features, self.g)

        _init_weights(self.w_x_uh, self.in_features, self.g)
        _init_weights(self.w_rh_uh, self.hidden_state_features, self.g)


class LSTMLayer(ContextAndStateDependentLayer):
    def __init__(self, in_features: int, hidden_state_features: int, g: torch.Generator=None):
        self.in_features = in_features
        self.hidden_state_features = hidden_state_features
        self.g = g

        # forget gate
        self.w_hf = torch.empty(hidden_state_features, hidden_state_features)
        self.w_xf = torch.empty(in_features, hidden_state_features)
        self.b_f = torch.zeros(hidden_state_features)

        # add context gate
        self.w_hg = torch.empty(hidden_state_features, hidden_state_features)
        self.w_xg = torch.empty(in_features, hidden_state_features)
        self.b_g = torch.zeros(hidden_state_features)

        # add context mask
        self.w_hi = torch.empty(hidden_state_features, hidden_state_features)
        self.w_xi = torch.empty(in_features, hidden_state_features)
        self.b_i = torch.zeros(hidden_state_features)

        # output
        self.w_ho = torch.empty(hidden_state_features, hidden_state_features)
        self.w_xo = torch.empty(in_features, hidden_state_features)
        self.b_o = torch.zeros(hidden_state_features)
        self._init_parameters()

    def __call__(self, x: torch.Tensor, hidden_state: torch.Tensor, context: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # forget gate
        f = torch.sigmoid(self.b_f + torch.matmul(x, self.w_xf) + torch.matmul(hidden_state, self.w_hf))
        # Hadamard product
        c_f = f * context

        # context gate
        g = torch.tanh(self.b_g + torch.matmul(x, self.w_xg) + torch.matmul(hidden_state, self.w_hg))
        # context mask
        i = torch.sigmoid(self.b_i + torch.matmul(x, self.w_xi) + torch.matmul(hidden_state, self.w_hi))
        addition_to_context = g * i

        new_context = c_f + addition_to_context

        # output
        o = torch.sigmoid(self.b_o + torch.matmul(x, self.w_xo) + torch.matmul(hidden_state, self.w_ho))
        new_hidden_state = o * torch.tanh(new_context)

        return new_hidden_state, new_hidden_state, new_context


    def parameters(self) -> List[torch.Tensor]:
        return [self.w_hf, self.w_xf, self.b_f,
                self.w_hg, self.w_xg, self.b_g,
                self.w_hi, self.w_xi, self.b_i,
                self.w_ho, self.w_xo, self.b_o]


    def _init_parameters(self) -> None:
        _init_weights(self.w_xf, self.in_features, self.g)
        _init_weights(self.w_xg, self.in_features, self.g)
        _init_weights(self.w_xi, self.in_features, self.g)
        _init_weights(self.w_xo, self.in_features, self.g)

        _init_weights(self.w_hf, self.hidden_state_features, self.g)
        _init_weights(self.w_hg, self.hidden_state_features, self.g)
        _init_weights(self.w_hi, self.hidden_state_features, self.g)
        _init_weights(self.w_ho, self.hidden_state_features, self.g)
