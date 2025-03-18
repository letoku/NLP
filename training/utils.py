from typing import List
from matplotlib import pyplot as plt


def plot_losses(training_stats):
    """
    Plots the training and validation losses over epochs.

    Args:
        training_stats: Dictionary returned by the train_model function containing 'train_loss' and 'val_loss'.
    """
    epochs = range(1, len(training_stats['train_loss']) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_stats['train_loss'], label='Training Loss', color='blue', linestyle='-', marker='o')
    plt.plot(epochs, training_stats['val_loss'], label='Validation Loss', color='red', linestyle='--', marker='x')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)

    plt.show()

def print_stats(epoch: int, num_epochs: int, train_losses: List[float], val_losses: List[float]) -> None:
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Train Loss: {train_losses[-1]:.4f}, "
        f"Val Loss: {val_losses[-1]:.4f}"
    )