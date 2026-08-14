import json
import fasttext

INPUT = "test_extracted.jsonl"
OUTPUT = "language_filtered.jsonl"
MODEL = "lid.176.bin"

# Languages we want to keep for INDAI
TARGET_LANGUAGES = {
    "en",  # English
    "hi",  # Hindi
    "bn",  # Bengali
    "ta",  # Tamil
    "te",  # Telugu
    "mr",  # Marathi
    "gu",  # Gujarati
    "kn",  # Kannada
    "ml",  # Malayalam
    "pa",  # Punjabi
    "ur",  # Urdu
    "or",  # Odia
    "as",  # Assamese
    "ne",  # Nepali
    "sa",  # Sanskrit
}

print("Loading FastText language model...")
model = fasttext.load_model(MODEL)

total = 0
kept = 0
removed = 0

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

        # FastText expects reasonable amounts of text
        sample = text[:5000].replace("\n", " ")

        labels, probabilities = model.predict(sample, k=1)

        language = labels[0].replace("__label__", "")
        confidence = probabilities[0]

        if language in TARGET_LANGUAGES and confidence >= 0.70:

            record["language"] = language
            record["language_confidence"] = round(float(confidence), 4)

            fout.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

            kept += 1

        else:
            removed += 1

        if total % 1000 == 0:
            print(
                f"Processed: {total:,} | "
                f"Kept: {kept:,} | "
                f"Removed: {removed:,}"
            )

print("\n==============================")
print("LANGUAGE FILTER COMPLETE")
print("==============================")
print(f"Total processed : {total:,}")
print(f"Kept            : {kept:,}")
print(f"Removed         : {removed:,}")
print(f"Output          : {OUTPUT}")