import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import ForwardModule

from .utils import print_stats


def optim_update_step(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def custom_update_step(model, loss, lr):
    for param in model.parameters:
        param.grad = None

    loss.backward()

    for param in model.parameters:
        param.data -= param.grad * lr


def train_loop(model: ForwardModule, train_loader: DataLoader, train_losses: list[float], lr=1e-3,
               optimizer=None) -> None:
    model.set_train_mode()
    running_train_loss = 0.0

    for x, y in train_loader:
        preds = model(x)
        loss = F.cross_entropy(preds, y)

        if optimizer is None:
            custom_update_step(model, loss, lr)
        else:
            optim_update_step(optimizer, loss)

        running_train_loss += loss.item()

    train_losses.append(running_train_loss / len(train_loader))


def val_loop(model: ForwardModule, val_loader: DataLoader, val_losses: list[float]) -> None:
    running_val_loss = 0.0

    model.set_eval_mode()
    for x, y in val_loader:
        preds = model(x)
        loss = F.cross_entropy(preds, y)
        running_val_loss += loss.item()

    val_losses.append(running_val_loss / len(val_loader))



def train_model(model: ForwardModule, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int,
                optimizer=None) -> dict:
    """
    Returns:
        A dictionary containing training statistics:
        'train_loss', 'val_loss'
    """
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        if epoch < 20:
            lr = 1e-1
        elif epoch < 40:
            lr = 1e-2
        elif epoch < 50:
            lr = 5e-3
        else:
            lr = 1e-3

        if optimizer is not None:
            train_loop(model, train_loader, train_losses, optimizer=optimizer)
        else:
            train_loop(model, train_loader, train_losses, lr=lr)
        val_loop(model, val_loader, val_losses)
        print_stats(epoch, num_epochs, train_losses, val_losses)

    return {
        'train_loss': train_losses,
        'val_loss': val_losses
    }