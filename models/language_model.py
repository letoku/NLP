import torch
from .module import Module

class LanguageModel(Module):
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
