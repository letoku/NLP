import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import StateDependentModule


def is_zero(x: float, eps: float = 1e-12) -> bool:
    if x < eps:
        return True
    else:
        return False

def optim_update_step(optimizer, losses, history_losses, tokens_n):
    running_train_loss = 0.0
    optimizer.zero_grad()
    for loss in losses:
        (loss.sum() / tokens_n).backward(retain_graph=True)
        running_train_loss += loss.sum().item() / tokens_n
    optimizer.step()

    history_losses.append(running_train_loss)


def custom_update_step(model, losses, history_losses, tokens_n, lr):
    running_train_loss = 0.0
    for param in model.parameters:
        param.grad = None

    for loss in losses:
        (loss.sum() / tokens_n).backward(retain_graph=True)
        running_train_loss += loss.sum().item() / tokens_n

    for param in model.parameters:
        param.data -= param.grad * lr

    history_losses.append(running_train_loss)


def train_loop(model: StateDependentModule, train_loader: DataLoader, tokens_n: int, train_losses: list[float], lr=1e-3, optimizer=None) -> None:
    hidden_units, number_of_hidden_layers = model.hidden_states_dimensions()
    model.set_train_mode()

    losses = []
    for x_seq, y_seq, w_seq in train_loader:
        hidden_states = torch.zeros(len(x_seq), hidden_units, number_of_hidden_layers)
        for i in range(x_seq.shape[1]):
            preds, hidden_states = model(x_seq[:, i], hidden_states)
            losses.append(F.cross_entropy(preds, y_seq[:, i], reduction='none') * w_seq[:, i])

    if optimizer is None:
        custom_update_step(model, losses, train_losses, tokens_n, lr)
    else:
        optim_update_step(optimizer, losses, train_losses, tokens_n)



def val_loop(model: StateDependentModule, val_loader: DataLoader, tokens_n: int, val_losses: list[float]) -> None:
    running_val_loss = 0.0
    hidden_units, number_of_hidden_layers = model.hidden_states_dimensions()
    model.set_eval_mode()

    for x_seq, y_seq, w_seq in val_loader:
        hidden_states = torch.zeros(len(x_seq), hidden_units, number_of_hidden_layers)
        for i in range(x_seq.shape[1]):
            preds, hidden_states = model(x_seq[:, i], hidden_states)
            loss = F.cross_entropy(preds, y_seq[:, i], reduction='none') * w_seq[:, i]
            running_val_loss += loss.sum().item() / tokens_n

    val_losses.append(running_val_loss)


def print_stats(epoch: int, num_epochs: int, train_losses: list[float], val_losses: list[float]) -> None:
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Train Loss: {train_losses[-1]:.4f}, "
        f"Val Loss: {val_losses[-1]:.4f}"
    )


def train_model(model: StateDependentModule,
                train_loader: DataLoader, train_tokens_n: int,
                val_loader: DataLoader, val_tokens_n: int,
                num_epochs: int, optimizer=None) -> dict:
    """
    Returns:
        A dictionary containing training statistics:
        'train_loss', 'val_loss'
    """
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        lr = 0.5
        if optimizer is not None:
            train_loop(model, train_loader, train_tokens_n, train_losses, optimizer=optimizer)
        else:
            train_loop(model, train_loader, train_tokens_n, train_losses, lr=lr)

        val_loop(model, val_loader, val_tokens_n, val_losses)
        print_stats(epoch, num_epochs, train_losses, val_losses)

    return {
        'train_loss': train_losses,
        'val_loss': val_losses
    }