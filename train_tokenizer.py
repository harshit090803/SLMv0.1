from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
import json

TRAIN_FILE = "train.jsonl"
TOKENIZER_FILE = "indai_tokenizer.json"

VOCAB_SIZE = 32000

print("================================")
print("INDAI TOKENIZER TRAINING")
print("================================")
print(f"Training file : {TRAIN_FILE}")
print(f"Vocabulary    : {VOCAB_SIZE}")

# Create BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="<unk>"))

# Byte-level preprocessing
tokenizer.pre_tokenizer = ByteLevel()

# Trainer
trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=[
        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>"
    ]
)

def text_iterator():
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                text = record.get("text", "").strip()

                if text:
                    yield text

            except json.JSONDecodeError:
                continue


print("\nTraining tokenizer...")
tokenizer.train_from_iterator(
    text_iterator(),
    trainer=trainer
)

# Save tokenizer
tokenizer.save(TOKENIZER_FILE)

print("\n================================")
print("TOKENIZER TRAINING COMPLETE")
print("================================")
print(f"Saved to      : {TOKENIZER_FILE}")
print(f"Vocabulary    : {tokenizer.get_vocab_size()}")

# Quick test
samples = [
    "Hello, how are you?",
    "India is a diverse country.",
    "भारत एक विविध देश है।",
    "Namaste bhai, kya haal hai?"
]

print("\nTOKENIZER TEST")
print("--------------------------------")

for text in samples:
    encoding = tokenizer.encode(text)

    print(f"\nText   : {text}")
    print(f"Tokens : {encoding.tokens}")
    print(f"IDs    : {encoding.ids}")