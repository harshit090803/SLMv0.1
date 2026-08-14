import json
import random

INPUT = "near_deduplicated.jsonl"

TRAIN_OUTPUT = "train.jsonl"
VAL_OUTPUT = "validation.jsonl"

VAL_RATIO = 0.05
SEED = 42

random.seed(SEED)

records = []

print("Loading cleaned dataset...")

with open(INPUT, "r", encoding="utf-8") as f:
    for line in f:
        try:
            record = json.loads(line)
            text = record.get("text", "").strip()

            if text:
                records.append(record)

        except json.JSONDecodeError:
            continue


print(f"Loaded: {len(records):,} documents")

# Shuffle deterministically
random.shuffle(records)

val_size = int(len(records) * VAL_RATIO)

validation = records[:val_size]
train = records[val_size:]

with open(TRAIN_OUTPUT, "w", encoding="utf-8") as f:
    for record in train:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )

with open(VAL_OUTPUT, "w", encoding="utf-8") as f:
    for record in validation:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )


print()
print("==============================")
print("DATASET SPLIT COMPLETE")
print("==============================")
print(f"Total      : {len(records):,}")
print(f"Training   : {len(train):,}")
print(f"Validation : {len(validation):,}")
print(f"Train %    : {len(train) / len(records) * 100:.2f}%")
print(f"Val %      : {len(validation) / len(records) * 100:.2f}%")