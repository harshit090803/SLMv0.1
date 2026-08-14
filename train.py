import os
import math
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import IndAI, IndAIConfig


# ============================================================
# INDAI sLLM v0.1
# TRAINING CONFIGURATION
# ============================================================

TRAIN_FILE = "train_sequences.pt"
VAL_FILE = "validation_sequences.pt"

CHECKPOINT_DIR = "checkpoints"
BEST_MODEL = os.path.join(CHECKPOINT_DIR, "indai_best.pt")
LATEST_MODEL = os.path.join(CHECKPOINT_DIR, "indai_latest.pt")

VOCAB_SIZE = 32000
SEQ_LEN = 2048

BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8

EPOCHS = 10

LEARNING_RATE = 3e-4
MIN_LR = 3e-5

WEIGHT_DECAY = 0.1

WARMUP_STEPS = 100

MAX_GRAD_NORM = 1.0

LOG_INTERVAL = 1
VALIDATE_INTERVAL = 50

NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


USE_AMP = DEVICE.type == "cuda"

print()
print("=" * 50)
print("INDAI sLLM v0.1")
print("TRAINING ENGINE")
print("=" * 50)

print()
print(f"Device          : {DEVICE}")

if DEVICE.type == "cuda":
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(
        f"VRAM            : "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

print(f"Batch size      : {BATCH_SIZE}")
print(f"Gradient accum. : {GRAD_ACCUM_STEPS}")
print(f"Effective batch : {BATCH_SIZE * GRAD_ACCUM_STEPS}")
print(f"Sequence length : {SEQ_LEN}")
print(f"Epochs          : {EPOCHS}")
print(f"Learning rate   : {LEARNING_RATE}")
print()


# ============================================================
# DATASET
# ============================================================

class SequenceDataset(Dataset):

    def __init__(self, path):

        print(f"Loading dataset: {path}")

        self.data = torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )

        if not isinstance(self.data, torch.Tensor):
            self.data = torch.tensor(self.data)

        self.data = self.data.long()

        print(f"Sequences       : {len(self.data):,}")
        print(f"Shape           : {tuple(self.data.shape)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sequence = self.data[idx]

        # Input:
        # token 0 ... token N-2
        #
        # Target:
        # token 1 ... token N-1

        x = sequence[:-1]
        y = sequence[1:]

        return x, y


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 50)
print("LOADING DATA")
print("=" * 50)

train_dataset = SequenceDataset(TRAIN_FILE)
val_dataset = SequenceDataset(VAL_FILE)

print()


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda")
)


# ============================================================
# MODEL
# ============================================================

# ============================================================
# BUILD MODEL
# ============================================================

print()
print("=" * 50)
print("BUILDING MODEL")
print("=" * 50)

# Create default INDAI configuration
config = IndAIConfig()

print()
print("MODEL CONFIG")
print("-" * 30)
print(f"Vocabulary size : {config.vocab_size:,}")
print(f"Context length  : {config.max_seq_len:,}")
print(f"Embedding size  : {config.d_model}")
print(f"Transformer layers : {config.n_layers}")
print(f"Attention heads : {config.n_heads}")
print(f"FFN size        : {config.d_ff}")
print(f"Dropout         : {config.dropout}")

# Build model
model = IndAI(config)

# Move model to CPU/GPU
model = model.to(DEVICE)

# Parameter count
total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print()
print("MODEL PARAMETERS")
print("-" * 30)
print(f"Total parameters     : {total_params:,}")
print(f"Trainable parameters : {trainable_params:,}")
print(f"Parameters (M)       : {total_params / 1e6:.2f}M")
print(f"Device               : {DEVICE}")

config = IndAIConfig()

model = IndAI(config)

model = model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print(f"Total parameters     : {total_params:,}")
print(f"Trainable parameters : {trainable_params:,}")
print(f"Device               : {DEVICE}")

model = model.to(DEVICE)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print()
print(f"Total parameters     : {total_params:,}")
print(f"Trainable parameters : {trainable_params:,}")
print(f"Parameters (M)       : {total_params / 1e6:.2f}M")
print()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING RATE SCHEDULE
# ============================================================

steps_per_epoch = math.ceil(
    len(train_loader) / GRAD_ACCUM_STEPS
)

total_steps = steps_per_epoch * EPOCHS


def get_lr(step):

    # Warmup
    if step < WARMUP_STEPS:

        return LEARNING_RATE * (
            (step + 1) / WARMUP_STEPS
        )

    # Progress after warmup
    progress = (
        step - WARMUP_STEPS
    ) / max(
        1,
        total_steps - WARMUP_STEPS
    )

    progress = min(progress, 1.0)

    # Cosine decay
    cosine = 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )

    return MIN_LR + (
        LEARNING_RATE - MIN_LR
    ) * cosine


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# MIXED PRECISION
# ============================================================

if USE_AMP:

    scaler = torch.amp.GradScaler("cuda")

else:

    scaler = None


# ============================================================
# CHECKPOINT DIRECTORY
# ============================================================

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate():

    model.eval()

    total_loss = 0.0
    total_batches = 0

    for x, y in val_loader:

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        if USE_AMP:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                logits, _ = model(x)

                loss = criterion(
                    logits.reshape(-1, VOCAB_SIZE),
                    y.reshape(-1)
                )

        else:

            logits, _ = model(x)

            loss = criterion(
                logits.reshape(-1, VOCAB_SIZE),
                y.reshape(-1)
            )

        total_loss += loss.item()
        total_batches += 1

    model.train()

    return total_loss / max(
        1,
        total_batches
    )


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    epoch,
    step,
    best_val_loss
):

    checkpoint = {

        "epoch": epoch,

        "step": step,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "best_val_loss":
            best_val_loss,

    }

    if scaler is not None:

        checkpoint[
            "scaler_state_dict"
        ] = scaler.state_dict()

    torch.save(
        checkpoint,
        path
    )
# ============================================================
# TRAINING
# ============================================================

print("=" * 50)
print("STARTING TRAINING")
print("=" * 50)

print()
print(f"Training sequences : {len(train_dataset):,}")
print(f"Validation seqs    : {len(val_dataset):,}")
print(f"Steps / epoch      : {steps_per_epoch:,}")
print(f"Total steps        : {total_steps:,}")
print()

global_step = 0
best_val_loss = float("inf")

# ============================================================
# TRAIN LOOP
# ============================================================

for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0.0
    epoch_start = time.time()

    optimizer.zero_grad(set_to_none=True)

    print()
    print("=" * 50)
    print(f"EPOCH {epoch + 1}/{EPOCHS}")
    print("=" * 50)

    for batch_idx, (x, y) in enumerate(train_loader):

        # ----------------------------------------------------
        # Move data
        # ----------------------------------------------------

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Learning rate
        # ----------------------------------------------------

        current_lr = get_lr(global_step)

        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # ----------------------------------------------------
        # Forward + Loss
        # ----------------------------------------------------

        if USE_AMP:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                logits, _ = model(x)

                loss = criterion(
                    logits.reshape(-1, VOCAB_SIZE),
                    y.reshape(-1)
                )

                raw_loss = loss.item()

                loss = (
                    loss /
                    GRAD_ACCUM_STEPS
                )

        else:

            logits, _ = model(x)

            loss = criterion(
                logits.reshape(-1, VOCAB_SIZE),
                y.reshape(-1)
            )

            raw_loss = loss.item()

            loss = (
                loss /
                GRAD_ACCUM_STEPS
            )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        if scaler is not None:

            scaler.scale(loss).backward()

        else:

            loss.backward()

        epoch_loss += raw_loss

        # ----------------------------------------------------
        # Optimizer step
        # ----------------------------------------------------

        should_step = (
            (batch_idx + 1) % GRAD_ACCUM_STEPS == 0
            or
            (batch_idx + 1) == len(train_loader)
        )

        if should_step:

            # Unscale before gradient clipping
            if scaler is not None:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM
            )

            # Optimizer update
            if scaler is not None:

                scaler.step(optimizer)
                scaler.update()

            else:

                optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            global_step += 1

            # ------------------------------------------------
            # Training progress
            # ------------------------------------------------

            if global_step % LOG_INTERVAL == 0:

                elapsed = (
                    time.time()
                    - epoch_start
                )

                avg_loss = (
                    epoch_loss /
                    (batch_idx + 1)
                )

                print(
                    f"Epoch {epoch + 1:2d} | "
                    f"Batch {batch_idx + 1:4d}/{len(train_loader)} | "
                    f"Step {global_step:5d} | "
                    f"Loss {avg_loss:.4f} | "
                    f"LR {current_lr:.2e} | "
                    f"Time {elapsed:.1f}s"
                )

            # ------------------------------------------------
            # Periodic validation
            # ------------------------------------------------

            if (
                global_step > 0
                and
                global_step % VALIDATE_INTERVAL == 0
            ):

                print()
                print("Running validation...")

                val_loss = evaluate()

                print(
                    f"Validation Loss : "
                    f"{val_loss:.4f}"
                )

                # --------------------------------------------
                # Save best model
                # --------------------------------------------

                if val_loss < best_val_loss:

                    best_val_loss = val_loss

                    save_checkpoint(
                        BEST_MODEL,
                        epoch,
                        global_step,
                        best_val_loss
                    )

                    print(
                        "🔥 New best model saved!"
                    )

                    print(
                        f"Best Val Loss : "
                        f"{best_val_loss:.4f}"
                    )

                # --------------------------------------------
                # Save latest model
                # --------------------------------------------

                save_checkpoint(
                    LATEST_MODEL,
                    epoch,
                    global_step,
                    best_val_loss
                )

                print()

    # ========================================================
    # END OF EPOCH
    # ========================================================

    epoch_loss = (
        epoch_loss /
        max(1, len(train_loader))
    )

    print()
    print("Running epoch validation...")

    val_loss = evaluate()

    epoch_time = (
        time.time()
        - epoch_start
    )

    print()
    print("=" * 50)
    print(
        f"EPOCH {epoch + 1} COMPLETE"
    )
    print("=" * 50)

    print(
        f"Train Loss : {epoch_loss:.4f}"
    )

    print(
        f"Val Loss   : {val_loss:.4f}"
    )

    print(
        f"Time       : {epoch_time:.1f}s"
    )

    print(
        f"Best Val   : {best_val_loss:.4f}"
    )

    # --------------------------------------------------------
    # Save latest checkpoint
    # --------------------------------------------------------

    save_checkpoint(
        LATEST_MODEL,
        epoch + 1,
        global_step,
        best_val_loss
    )

    # --------------------------------------------------------
    # Save best checkpoint
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        save_checkpoint(
            BEST_MODEL,
            epoch + 1,
            global_step,
            best_val_loss
        )

        print(
            "🔥 New best model saved!"
        )


# ============================================================
# TRAINING COMPLETE
# ============================================================

print()
print("=" * 50)
print("🔥 INDAI TRAINING COMPLETE 🔥")
print("=" * 50)

print()
print(
    f"Final training step : "
    f"{global_step}"
)

print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)

print()
print(
    f"Best model          : "
    f"{BEST_MODEL}"
)

print(
    f"Latest checkpoint   : "
    f"{LATEST_MODEL}"
)

print()
print("INDAI sLLM v0.1 has been trained.")
print("=" * 50)