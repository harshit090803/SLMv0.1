import json
import os
from tokenizers import Tokenizer

# ==============================
# CONFIGURATION
# ==============================

TOKENIZER_FILE = "indai_tokenizer.json"

TRAIN_INPUT = "train.jsonl"
VAL_INPUT = "validation.jsonl"

TRAIN_OUTPUT = "train_tokens.jsonl"
VAL_OUTPUT = "validation_tokens.jsonl"

# Maximum number of tokens in one training sequence
MAX_LENGTH = 2048


# ==============================
# LOAD TOKENIZER
# ==============================

print("================================")
print("INDAI DATASET TOKENIZATION")
print("================================")

print("\nLoading INDAI tokenizer...")

tokenizer = Tokenizer.from_file(TOKENIZER_FILE)

print(f"Vocabulary size : {tokenizer.get_vocab_size()}")
print(f"Max sequence    : {MAX_LENGTH}")


# ==============================
# SPECIAL TOKENS
# ==============================

BOS_ID = tokenizer.token_to_id("<bos>")
EOS_ID = tokenizer.token_to_id("<eos>")

PAD_ID = tokenizer.token_to_id("<pad>")
UNK_ID = tokenizer.token_to_id("<unk>")

print("\nSpecial tokens:")
print(f"<pad> : {PAD_ID}")
print(f"<unk> : {UNK_ID}")
print(f"<bos> : {BOS_ID}")
print(f"<eos> : {EOS_ID}")


# ==============================
# TOKENIZE ONE FILE
# ==============================

def tokenize_file(input_file, output_file):

    total = 0
    processed = 0
    skipped = 0
    total_tokens = 0

    print("\n--------------------------------")
    print(f"Processing: {input_file}")
    print("--------------------------------")

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:

            total += 1

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                skipped += 1
                continue

            text = record.get("text", "").strip()

            if not text:
                skipped += 1
                continue

            # Tokenize
            encoding = tokenizer.encode(text)

            token_ids = encoding.ids

            # Add BOS and EOS
            if BOS_ID is not None:
                token_ids.insert(0, BOS_ID)

            if EOS_ID is not None:
                token_ids.append(EOS_ID)

            # Skip extremely long documents for now
            if len(token_ids) > MAX_LENGTH:
                token_ids = token_ids[:MAX_LENGTH]

                # Make sure sequence ends with EOS
                if EOS_ID is not None:
                    token_ids[-1] = EOS_ID

            output_record = {
                "input_ids": token_ids,
                "length": len(token_ids)
            }

            fout.write(
                json.dumps(
                    output_record,
                    ensure_ascii=False
                ) + "\n"
            )

            processed += 1
            total_tokens += len(token_ids)

            # Progress
            if processed % 500 == 0:
                print(
                    f"Processed: {processed:,} | "
                    f"Tokens: {total_tokens:,}"
                )

    print("\nCompleted:")
    print(f"Input records : {total:,}")
    print(f"Processed     : {processed:,}")
    print(f"Skipped       : {skipped:,}")
    print(f"Total tokens  : {total_tokens:,}")

    if processed > 0:
        print(
            f"Average tokens/document : "
            f"{total_tokens / processed:.2f}"
        )

    print(f"Output        : {output_file}")


# ==============================
# TRAIN DATA
# ==============================

tokenize_file(
    TRAIN_INPUT,
    TRAIN_OUTPUT
)


# ==============================
# VALIDATION DATA
# ==============================

tokenize_file(
    VAL_INPUT,
    VAL_OUTPUT
)


# ==============================
# FINAL SUMMARY
# ==============================

print("\n================================")
print("TOKENIZATION COMPLETE")
print("================================")

print(f"Training output   : {TRAIN_OUTPUT}")
print(f"Validation output : {VAL_OUTPUT}")

print("\nINDAI dataset is now represented as token IDs.")
print("Next step: build the training sequence/batching pipeline.")