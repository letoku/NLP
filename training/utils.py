from typing import List
from matplotlib import pyplot as plt
import mlflow.pyfunc
import mlflow


def plot_losses(training_stats, save_path: str = None):
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
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def print_stats(epoch: int, num_epochs: int, train_losses: List[float], val_losses: List[float]) -> None:
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Train Loss: {train_losses[-1]:.4f}, "
        f"Val Loss: {val_losses[-1]:.4f}"
    )

def mlflow_save_run(experiment: str, model, train_loss: float, val_loss: float,
                    params: dict, server_uri: str, plot_path: str=None) -> None:
    wrapped_model =  _MlflowModelWrapper(model)
    mlflow.set_tracking_uri(uri=server_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run():
        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("val_loss", val_loss)
        mlflow.log_params(params)
        mlflow.pyfunc.log_model(artifact_path="model", python_model=wrapped_model)
        if plot_path is not None:
            mlflow.log_artifact(plot_path)


class _MlflowModelWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, model):
      self.model = model

  def predict(self, context, model_input, params=None):
      return 0
