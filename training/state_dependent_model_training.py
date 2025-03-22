from typing import List, Dict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import StateDependentModule
from models.module import ContextAndStateDependentModule

from .utils import print_stats


def optim_update_step(optimizer, losses, tokens_n):
    optimizer.zero_grad()
    for loss in losses:
        (loss.sum() / tokens_n).backward(retain_graph=True)
    optimizer.step()


def custom_update_step(model, losses, tokens_n, lr):
    for param in model.parameters:
        param.grad = None

    for loss in losses:
        (loss.sum() / tokens_n).backward(retain_graph=True)

    for param in model.parameters:
        if param.grad is not None:
            param.data -= param.grad * lr


def train_loop(model: StateDependentModule | ContextAndStateDependentModule, train_loader: DataLoader,
               train_losses: List[float], lr=1e-3, optimizer=None, device: torch.device=None) -> None:
    if device is None:
        device = torch.device('cpu')
    context_and_state_mode = True if isinstance(model, ContextAndStateDependentModule) else False

    hidden_units, number_of_hidden_layers = model.get_hidden_states_dims()
    model.set_train_mode()

    running_train_loss = 0.0
    all_tokens = 0
    for x_seq, y_seq, w_seq in train_loader:
        losses = []
        tokens = w_seq.sum().sum().item()
        all_tokens += tokens
        hidden_states = torch.zeros(len(x_seq), hidden_units, number_of_hidden_layers).to(device)
        if context_and_state_mode:
            context = torch.zeros(len(x_seq), hidden_units, number_of_hidden_layers).to(device)

        for i in range(x_seq.shape[1]):
            i_device = torch.tensor(i).to(device)
            if context_and_state_mode:
                preds, hidden_states, context = model(x_seq[:, i_device], hidden_states, context)
            else:
                preds, hidden_states = model(x_seq[:, i_device], hidden_states)
            losses.append(F.cross_entropy(preds, y_seq[:, i_device], reduction='none') * w_seq[:, i_device])
            running_train_loss += losses[-1].sum().item()

        if optimizer is None:
            custom_update_step(model, losses, tokens, lr)
        else:
            optim_update_step(optimizer, losses, tokens)

    train_losses.append(running_train_loss / all_tokens)


def val_loop(model: StateDependentModule | ContextAndStateDependentModule,
             val_loader: DataLoader, val_losses: List[float], device: torch.device=None) -> None:
    if device is None:
        device = torch.device('cpu')
    context_and_state_mode = True if isinstance(model, ContextAndStateDependentModule) else False
    running_val_loss = 0.0
    hidden_units, number_of_hidden_layers = model.get_hidden_states_dims()
    model.set_eval_mode()
    all_tokens = 0
    for x_seq, y_seq, w_seq in val_loader:
        tokens = w_seq.sum().sum().item()
        all_tokens += tokens
        hidden_states = torch.zeros(len(x_seq), hidden_units, number_of_hidden_layers).to(device)
        if context_and_state_mode:
            context = torch.zeros(len(x_seq), hidden_units, number_of_hidden_layers).to(device)

        for i in range(x_seq.shape[1]):
            if context_and_state_mode:
                preds, hidden_states, context = model(x_seq[:, i], hidden_states, context)
            else:
                preds, hidden_states = model(x_seq[:, i], hidden_states)
            loss = F.cross_entropy(preds, y_seq[:, i], reduction='none') * w_seq[:, i]
            running_val_loss += loss.sum().item()

    val_losses.append(running_val_loss / all_tokens)


def train_model(model: StateDependentModule | ContextAndStateDependentModule,
                train_loader: DataLoader,
                val_loader: DataLoader,
                num_epochs: int, optimizer=None, lr=1e-3, device: torch.device=None) -> Dict[str, List[float]]:
    """
    Returns:
        A dictionary containing training statistics:
        'train_loss', 'val_loss'
    """
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.set_train_mode()
        if optimizer is not None:
            train_loop(model, train_loader, train_losses, optimizer=optimizer, device=device)
        else:
            train_loop(model, train_loader, train_losses, lr=lr, device=device)

        model.set_eval_mode()
        val_loop(model, val_loader, val_losses, device=device)
        print_stats(epoch, num_epochs, train_losses, val_losses)

    return {
        'train_loss': train_losses,
        'val_loss': val_losses
    }