import torch
from .module import ForwardModule, StateDependentModule


class LanguageModel(ForwardModule):
    def __init__(self, specs: list[tuple[str, dict]], block_size: int, itos: dict[int, str], g: torch.Generator=None):
        super().__init__(specs, g)
        self.itos = itos
        self.block_size = block_size

    def sample(self):
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


class StateDependentLanguageModel(StateDependentModule):
    def __init__(self, specs: list[tuple[str, dict]], hidden_units: int, itos: dict[int, str], g: torch.Generator=None):
        super().__init__(specs, hidden_units, g)
        self.itos = itos

    def sample(self):
        self.set_eval_mode()
        context = torch.tensor([0])
        outputs = []
        hidden_units, number_of_hidden_layers = self.hidden_states_dimensions()
        hidden_states = torch.zeros(1, hidden_units, number_of_hidden_layers)
        while True:
            probs, hidden_states = self.predict_proba(torch.unsqueeze(context, dim=0), hidden_states)
            out = torch.multinomial(probs, num_samples=1, replacement=True)[0]
            outputs.append(out.item())
            if out.item() == 0:
                break
            else:
                context = out

        return ''.join(self.itos[o] for o in outputs)
