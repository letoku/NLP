from typing import List, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from .utils import print_stats


def train_loop(model: nn.Module, train_loader: DataLoader, train_losses: List[float], all_tokens: int, optimizer) -> None:
    running_train_loss = 0.0
    for inputs, targets, lengths in train_loader:
        outputs = model(inputs, lengths)

        _, _, vocab_size = outputs.size()
        outputs_flat = outputs.view(-1, vocab_size)
        targets_flat = targets.view(-1)

        loss = cross_entropy(outputs_flat, targets_flat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_train_loss += cross_entropy(outputs_flat, targets_flat, reduction="sum").item()

    train_losses.append(running_train_loss/all_tokens)


def val_loop(model: nn.Module, val_loader: DataLoader, val_losses: List[float], all_tokens: int) -> None:
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, targets, lengths in val_loader:
            outputs = model(inputs, lengths)

            _, _, vocab_size = outputs.size()
            outputs_flat = outputs.view(-1, vocab_size)
            targets_flat = targets.view(-1)

            loss = cross_entropy(outputs_flat, targets_flat, reduction="sum")
            running_val_loss += loss.item()

    val_losses.append(running_val_loss/all_tokens)


def train_model(model: nn.Module,
                train_loader: DataLoader,
                train_tokens: int,
                val_loader: DataLoader,
                val_tokens: int,
                num_epochs: int, optimizer) -> Dict[str, List[float]]:
    """
    Returns:
        A dictionary containing training statistics:
        'train_loss', 'val_loss'
    """
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loop(model, train_loader, train_losses, train_tokens, optimizer)

        val_loop(model, val_loader, val_losses, val_tokens)
        print_stats(epoch, num_epochs, train_losses, val_losses)

    return {
        'train_loss': train_losses,
        'val_loss': val_losses
    }
