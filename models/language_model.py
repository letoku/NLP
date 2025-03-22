from typing import List, Tuple, Dict
import torch
from abc import ABC, abstractmethod

from .module import ForwardModule, StateDependentModule, ContextAndStateDependentModule


class LanguageModel(ABC):
    @abstractmethod
    def sample(self) -> str:
        pass


class ContextWindowLanguageModel(ForwardModule, LanguageModel):
    def __init__(self, specs: List[Tuple[str, Dict]], block_size: int, itos: Dict[int, str], g: torch.Generator=None):
        super().__init__(specs, g)
        self.itos = itos
        self.block_size = block_size

    def sample(self) -> str:
        self.set_eval_mode()
        context = torch.tensor([0] * self.block_size)
        outputs = []
        while True:
            probs = self.predict_proba(torch.unsqueeze(context, dim=0))
            out = torch.multinomial(probs, num_samples=1, replacement=True)[0]
            outputs.append(out.item())
            if out.item() == 0:
                break
            else:
                context = torch.cat((context[1:], out), dim=0)

        return ''.join(self.itos[o] for o in outputs)


class StateDependentLanguageModel(StateDependentModule, LanguageModel):
    def __init__(self, specs: List[Tuple[str, Dict]], hidden_units: int, itos: Dict[int, str], g: torch.Generator=None):
        super().__init__(specs, hidden_units, g)
        self.itos = itos

    def sample(self, device: torch.device=None) -> str:
        if device is None:
            device = torch.device('cpu')
        self.set_eval_mode()
        context = torch.tensor([0]).to(device)
        outputs = []
        hidden_units, number_of_hidden_layers = self.get_hidden_states_dims()
        hidden_states = torch.zeros(1, hidden_units, number_of_hidden_layers).to(device)
        while True:
            probs, hidden_states = self.predict_proba(torch.unsqueeze(context, dim=0), hidden_states)
            out = torch.multinomial(probs, num_samples=1, replacement=True)[0]
            outputs.append(out.item())
            if out.item() == 0:
                break
            else:
                context = out

        return ''.join(self.itos[o] for o in outputs)


class ContextAndStateDependentLanguageModel(ContextAndStateDependentModule, LanguageModel):
    def __init__(self, specs: List[Tuple[str, Dict]], hidden_units: int, itos: Dict[int, str], g: torch.Generator=None):
        super().__init__(specs, hidden_units, g)
        self.itos = itos

    def sample(self, device: torch.device=None) -> str:
        if device is None:
            device = torch.device('cpu')
        self.set_eval_mode()
        inp = torch.tensor([0]).to(device)
        outputs = []
        hidden_units, number_of_hidden_layers = self.get_hidden_states_dims()
        hidden_states = torch.zeros(1, hidden_units, number_of_hidden_layers).to(device)
        context = torch.zeros(1, hidden_units, number_of_hidden_layers).to(device)
        while True:
            probs, hidden_states, context = self.predict_proba(torch.unsqueeze(inp, dim=0), hidden_states, context)
            out = torch.multinomial(probs, num_samples=1, replacement=True)[0]
            outputs.append(out.item())
            if out.item() == 0:
                break
            else:
                inp = out

        return ''.join(self.itos[o] for o in outputs)
