import json
import hashlib

INPUT = "quality_filtered.jsonl"
OUTPUT = "deduplicated.jsonl"

seen = set()

total = 0
kept = 0
removed = 0

print("Starting exact deduplication...")

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

        # Normalize whitespace
        normalized = " ".join(text.split())

        # SHA-256 fingerprint
        fingerprint = hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

        if fingerprint in seen:
            removed += 1
            continue

        seen.add(fingerprint)

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
                f"Removed: {removed:,}"
            )

print()
print("==============================")
print("EXACT DEDUPLICATION COMPLETE")
print("==============================")
print(f"Processed : {total:,}")
print(f"Kept      : {kept:,}")
print(f"Removed   : {removed:,}")
print(f"Output    : {OUTPUT}")