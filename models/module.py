from .layers import *
import torch.nn.functional as F


ALLOWED_LAYERS = {
    "Linear": Linear,
    "ReLU": ReLU,
    "Tanh": Tanh,
    "Embedding": Embedding
}


class Module:
    def __init__(self, specs: list[tuple[str, dict]], g: torch.Generator=None):
        self.layers = []
        self.g = g
        for spec in specs:
            layer_type, kwargs = spec
            assert layer_type in ALLOWED_LAYERS
            added_layer = ALLOWED_LAYERS[layer_type]
            self.layers.append(added_layer(**kwargs))

        self.parameters = []
        for layer in self.layers:
            self.parameters += layer.parameters()

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

