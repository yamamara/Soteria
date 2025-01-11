import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import optuna
from ultralytics import YOLO

# Configuration Parameters
DATA_YAML_PATH = "/home/artemis/Downloads/Gun Dataset/Gun with webcam views.v1i.yolov8/data.yaml"  # Path to your dataset configuration file
SAVE_DIR = "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
YOLO_MODEL_PATH = "yolo11x.pt"  # Path to your YOLOv11 model

# Verify dataset path exists
if not os.path.exists(DATA_YAML_PATH):
    raise FileNotFoundError(f"Dataset configuration file not found: {DATA_YAML_PATH}")

# YOLO Training Wrapper
def yolo_train(epochs, batch_size, learning_rate):
    """Train YOLO model and return validation metrics."""
    # Load YOLO model from file
    model = YOLO(YOLO_MODEL_PATH).to(DEVICE)

    # Explicitly set dataset path to avoid fallback to default
    model.data = DATA_YAML_PATH

    # Set hyperparameters
    model.overrides["lr0"] = learning_rate
    model.overrides["batch"] = batch_size  # Ensure batch_size is passed correctly

    train_writer = SummaryWriter(os.path.join(SAVE_DIR, "logs/train"))
    val_writer = SummaryWriter(os.path.join(SAVE_DIR, "logs/val"))

    # Early stopping parameters
    early_stopping_patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        try:
            # Train model for one epoch
            results = model.train(
                data=DATA_YAML_PATH,
                epochs=1,  # Train one epoch at a time to log metrics
                batch=batch_size,  # Pass batch size correctly here
                imgsz=416,
                device=DEVICE,
                lr0=learning_rate,
                project=SAVE_DIR,
                name="yolo_tuning"
            )

            # Extract metrics directly
            model.val()
            train_loss = results["train/loss"]
            val_loss = results["val/loss"]

            # Log metrics to TensorBoard
            train_writer.add_scalar("Loss/Epoch", train_loss, epoch)
            val_writer.add_scalar("Validation/Epoch", val_loss, epoch)

            print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        except torch.cuda.OutOfMemoryError:
            print("CUDA out of memory error encountered. Reducing batch size and retrying...")
            batch_size = max(1, batch_size // 2)  # Halve the batch size to free up memory
            model.overrides["batch"] = batch_size

    train_writer.close()
    val_writer.close()
    return best_val_loss

# Optuna Tuning Function
def objective(trial):
    """Objective function for Optuna."""
    epochs = trial.suggest_int("epochs", 20, 150)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)

    return yolo_train(epochs, batch_size, learning_rate)

# Run Optuna Study
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    print("Best hyperparameters:", study.best_params)
