import json
import re
from datasketch import MinHash, MinHashLSH

INPUT = "deduplicated.jsonl"
OUTPUT = "near_deduplicated.jsonl"

# Similarity threshold
# 0.90 = very conservative
THRESHOLD = 0.90

NUM_PERM = 128

lsh = MinHashLSH(
    threshold=THRESHOLD,
    num_perm=NUM_PERM
)

total = 0
kept = 0
removed = 0


def normalize(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def create_minhash(text):
    m = MinHash(num_perm=NUM_PERM)

    words = normalize(text).split()

    # 5-word shingles
    if len(words) < 5:
        shingles = [" ".join(words)]
    else:
        shingles = [
            " ".join(words[i:i + 5])
            for i in range(len(words) - 4)
        ]

    for shingle in shingles:
        m.update(shingle.encode("utf-8"))

    return m


print("Starting near-duplicate detection...")
print(f"Similarity threshold: {THRESHOLD}")
print()


with open(INPUT, "r", encoding="utf-8") as fin, \
     open(OUTPUT, "w", encoding="utf-8") as fout:

    for line in fin:

        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        text = record.get("text", "").strip()

        if not text:
            continue

        total += 1

        minhash = create_minhash(text)

        # Search for similar documents
        candidates = lsh.query(minhash)

        if candidates:
            removed += 1
            continue

        doc_id = f"doc_{kept}"

        lsh.insert(doc_id, minhash)

        fout.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )

        kept += 1

        if total % 500 == 0:
            print(
                f"Processed: {total:,} | "
                f"Kept: {kept:,} | "
                f"Near-duplicates removed: {removed:,}"
            )


print()
print("==============================")
print("NEAR-DEDUPLICATION COMPLETE")
print("==============================")
print(f"Processed : {total:,}")
print(f"Kept      : {kept:,}")
print(f"Removed   : {removed:,}")
print(f"Output    : {OUTPUT}")