import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
from src.models.model1 import config


# ============================================================
# ARCHITECTURE
# ============================================================

class SpectrogramModel(nn.Module):
    """
    Input:  (batch, 1, 200, 128)
    Output: (batch, 3)  — femininity, masculinity, atypicality ∈ [0, 1] (after sigmoiding)
    """
    def __init__(self, dropout: float):
        super().__init__()

        self.cnn = nn.Sequential(

            # Block 1 — (1, 200, 128) → (16, 100, 64)
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2 — (16, 100, 64) → (32, 50, 32)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3 — (32, 50, 32) → (64, 25, 16)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Collapse (64, 25, 16) → (64, 1, 4) = 256 values
        self.pool = nn.AdaptiveAvgPool2d((1, 4))

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 32),  # 64 channels * 1 * 4 = 256
            nn.ReLU(inplace=True),
            
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # (batch, 256)
        x = self.fc(x)
        return x


# ============================================================
# HELPERS
# ============================================================

def _get_device() -> tuple[torch.device, str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, device.type


def _make_dataloader(tensor_input: torch.Tensor, tensor_target: torch.Tensor, shuffle: bool = True) -> DataLoader:
    dataset = TensorDataset(tensor_input, tensor_target)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available()
    )


def _save_model(model: nn.Module, suffix: str) -> None:
    save_dir = os.path.join("misc/models", config.MODEL_NAME, "trained")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{suffix}.pth")
    torch.save(model.state_dict(), path)
    print(f"  Saved → {path}")


def _get_weights_path(custom_path: str) -> str:
    if custom_path:
        return custom_path
    return os.path.join("misc/models", config.MODEL_NAME, "model.pth")


# ============================================================
# TRAIN
# ============================================================

def train(
    tensor_input:    torch.Tensor,
    tensor_target:   torch.Tensor,
    tensor_input_dev:  torch.Tensor,
    tensor_target_dev: torch.Tensor,
    custom_weights_path:  str = ""
) -> None:

    device, device_type = _get_device()
    print(f"Device: {device}")

    # ── Prepare data ─────────────────────────────────────────
    x     = torch.tensor(tensor_input,      dtype=torch.float32).unsqueeze(1)
    y     = torch.tensor(tensor_target,     dtype=torch.float32)
    x_dev = torch.tensor(tensor_input_dev,  dtype=torch.float32).unsqueeze(1)
    y_dev = torch.tensor(tensor_target_dev, dtype=torch.float32)

    train_loader = _make_dataloader(x, y, shuffle=True)
    dev_loader   = _make_dataloader(x_dev, y_dev, shuffle=False)
    
    # ── Log Data ────────────────────────────────────────────────
    print(f"== Data Info ==")
    print(f"Train samples: {len(x)}")
    print(f"Dev samples:   {len(x_dev)}")
    print(f"Unique labels: {len(set(map(tuple, tensor_target.tolist())))}")

    # ── Model ─────────────────────────────────────────────────
    model = SpectrogramModel(config.dropout).to(device)

    # ── Load custom weights if provided ───────────────────────
    if custom_weights_path:
        if os.path.exists(custom_weights_path):
            print(f">> Loading weights from: {custom_weights_path} <<")
            state_dict = torch.load(custom_weights_path, map_location=device)
            model.load_state_dict(state_dict)
        else:
            print(f"WARNING: Weights path not found: {custom_weights_path}")
            exit(1)
    else:
        print(f">> No custom weights path provided, starting from the random values <<")
      
    # ── Setting up the optimizers ───────────────────────
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scaler    = GradScaler(device_type)
    best_dev_loss = float("inf")

    # ── Training loop ─────────────────────────────────────────
    print(f"== Training loop ==")
    for epoch in range(1, config.max_training_epochs + 1):

        # -- Train --
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            with autocast(device_type):
                output = model(batch_x)
                loss   = criterion(output, batch_y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # -- Validate --
        model.eval()
        dev_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in dev_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                with autocast(device_type):
                    output   = model(batch_x)
                    dev_loss += criterion(output, batch_y).item()

        avg_dev_loss = dev_loss / len(dev_loader)

        # -- Log --
        if epoch % config.training_log_step_epoch == 0:
            print(
                f"Epoch [{epoch:>5}/{config.max_training_epochs}]  "
                f"Train Loss: {avg_train_loss:.8f}  "
                f"Dev Loss:   {avg_dev_loss:.8f}"
            )

        # -- Checkpoint --
        if epoch % config.training_save_step_epoch == 0:
            _save_model(model, f"model_ep{epoch}_lossT{avg_train_loss:.8f}_lossD{avg_dev_loss:.8f}")

        # -- Best model --
        if avg_dev_loss < best_dev_loss:
            best_dev_loss = avg_dev_loss
            _save_model(model, f"model_best")
            print(f"  New best dev loss: ep{epoch} lossT{avg_train_loss:.8f} lossD{avg_dev_loss:.8f}")

    # -- Final save --
    _save_model(model, "model_final")


# ============================================================
# PREDICT
# ============================================================


def predict(
    tensor_input:         torch.Tensor,
    custom_weights_path:  str = "",
) -> torch.Tensor:
    """
    tensor_input — (batch, 200, 128), numpy array or torch tensor
    Returns      — (batch, 3) numpy array: femininity, masculinity, atypicality
    """
    x = torch.tensor(tensor_input, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(x, batch_size=config.batch_size,shuffle=False)

    model = SpectrogramModel(config.dropout)
    weights_path = _get_weights_path(custom_weights_path)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    
    preds = []

    with torch.no_grad():
        for batch_x in loader:
            prediction = torch.sigmoid(model(batch_x))
            preds.append(prediction)

    return torch.cat(preds, dim=0).numpy()


# ============================================================
# EXPORT
# ============================================================

def export_to_torchscript(weights_path: str, output_path: str) -> None:
    model = SpectrogramModel()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    example = torch.zeros(1, 1, 200, 128)
    traced = torch.jit.trace(model, example)
    traced.save(output_path)
    print(f"Exported → {output_path}")


