import json
import os
import torch
from torch.utils.data import Dataset, DataLoader

# ============================================================
# INDAI TRAINING SEQUENCE PIPELINE
# ============================================================

TRAIN_INPUT = "train_tokens.jsonl"
VAL_INPUT = "validation_tokens.jsonl"

TRAIN_OUTPUT = "train_sequences.pt"
VAL_OUTPUT = "validation_sequences.pt"

SEQ_LEN = 2048

BATCH_SIZE = 4

# ------------------------------------------------------------
# Load token IDs from JSONL
# ------------------------------------------------------------

def load_tokens(filename):
    tokens = []

    print(f"Loading: {filename}")

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            ids = record.get("input_ids", [])

            if not ids:
                continue

            tokens.extend(ids)

    print(f"Loaded tokens: {len(tokens):,}")

    return tokens


# ------------------------------------------------------------
# Create next-token prediction sequences
# ------------------------------------------------------------

def create_sequences(tokens, seq_len):

    sequences = []

    total_possible = len(tokens) // seq_len

    print(f"Creating sequences...")
    print(f"Sequence length : {seq_len}")
    print(f"Possible chunks : {total_possible:,}")

    for i in range(0, len(tokens) - seq_len, seq_len):

        chunk = tokens[i:i + seq_len + 1]

        if len(chunk) < seq_len + 1:
            break

        sequences.append(chunk)

    return sequences


# ------------------------------------------------------------
# Save sequences
# ------------------------------------------------------------

def save_sequences(sequences, filename):

    tensor = torch.tensor(
        sequences,
        dtype=torch.long
    )

    torch.save(tensor, filename)

    print(f"Saved: {filename}")
    print(f"Shape: {tuple(tensor.shape)}")


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class IndAIDataset(Dataset):

    def __init__(self, tensor):

        self.data = tensor

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        sequence = self.data[index]

        # Input
        x = sequence[:-1]

        # Target = next token
        y = sequence[1:]

        return x, y


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 40)
    print("INDAI TRAINING SEQUENCE PIPELINE")
    print("=" * 40)
    print()

    # --------------------------------------------------------
    # TRAINING DATA
    # --------------------------------------------------------

    train_tokens = load_tokens(TRAIN_INPUT)

    print()

    train_sequences = create_sequences(
        train_tokens,
        SEQ_LEN
    )

    print(
        f"Training sequences : {len(train_sequences):,}"
    )

    print()

    save_sequences(
        train_sequences,
        TRAIN_OUTPUT
    )

    print()

    # --------------------------------------------------------
    # VALIDATION DATA
    # --------------------------------------------------------

    val_tokens = load_tokens(VAL_INPUT)

    print()

    val_sequences = create_sequences(
        val_tokens,
        SEQ_LEN
    )

    print(
        f"Validation sequences : {len(val_sequences):,}"
    )

    print()

    save_sequences(
        val_sequences,
        VAL_OUTPUT
    )

    # --------------------------------------------------------
    # TEST DATASET
    # --------------------------------------------------------

    train_tensor = torch.load(
        TRAIN_OUTPUT,
        weights_only=True
    )

    val_tensor = torch.load(
        VAL_OUTPUT,
        weights_only=True
    )

    train_dataset = IndAIDataset(train_tensor)
    val_dataset = IndAIDataset(val_tensor)

    print()
    print("=" * 40)
    print("DATASET PIPELINE READY")
    print("=" * 40)

    print(
        f"Training sequences   : {len(train_dataset):,}"
    )

    print(
        f"Validation sequences : {len(val_dataset):,}"
    )

    print(
        f"Sequence length      : {SEQ_LEN}"
    )

    print(
        f"Batch size           : {BATCH_SIZE}"
    )

    # --------------------------------------------------------
    # DataLoader test
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    x, y = next(iter(train_loader))

    print()
    print("BATCH TEST")
    print("-" * 30)

    print(f"Input shape  : {x.shape}")
    print(f"Target shape : {y.shape}")

    print()
    print("Example input IDs:")
    print(x[0][:20].tolist())

    print()
    print("Example target IDs:")
    print(y[0][:20].tolist())

    print()
    print("=" * 40)
    print("INDAI SEQUENCE PIPELINE COMPLETE")
    print("=" * 40)