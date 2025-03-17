import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import ForwardModule, StateDependentModule, LanguageModel
import matplotlib.pyplot as plt


def plot_losses(training_stats):
    """
    Plots the training and validation losses over epochs.

    Args:
        training_stats: Dictionary returned by the train_model function containing 'train_loss' and 'val_loss'.
    """
    epochs = range(1, len(training_stats['train_loss']) + 1)

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_stats['train_loss'], label='Training Loss', color='blue', linestyle='-', marker='o')
    plt.plot(epochs, training_stats['val_loss'], label='Validation Loss', color='red', linestyle='--', marker='x')

    # Labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()


def train_loop(model: ForwardModule, train_loader: DataLoader, train_losses: list[float], lr=1e-3) -> None:
    model.set_train_mode()
    running_train_loss = 0.0

    for x, y in train_loader:
        preds = model(x)
        loss = F.cross_entropy(preds, y)

        for param in model.parameters:
            param.grad = None
        loss.backward()

        for param in model.parameters:
            param.data -= param.grad * lr

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


def print_stats(epoch: int, num_epochs: int, train_losses: list[float], val_losses: list[float]) -> None:
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Train Loss: {train_losses[-1]:.4f}, "
        f"Val Loss: {val_losses[-1]:.4f}"
    )


def train_model(model: ForwardModule, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int) -> dict:
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

        train_loop(model, train_loader, train_losses, lr)
        val_loop(model, val_loader, val_losses)
        print_stats(epoch, num_epochs, train_losses, val_losses)

    return {
        'train_loss': train_losses,
        'val_loss': val_losses
    }