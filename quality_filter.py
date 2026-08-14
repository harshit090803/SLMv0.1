import json
import re
import unicodedata

INPUT = "language_filtered.jsonl"
OUTPUT = "quality_filtered.jsonl"

MIN_CHARS = 200
MAX_CHARS = 500_000

# Obvious low-quality / junk indicators
SPAM_PATTERNS = [
    r"buy now",
    r"click here",
    r"free download",
    r"casino",
    r"jackpot",
    r"porn",
    r"xxx",
    r"sex cam",
    r"adult video",
    r"viagra",
    r"crypto giveaway",
    r"telegram",
    r"whatsapp.*contact",
    r"domain.*for sale",
]

URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)

def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)

    # Remove excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def repetition_ratio(text):
    """
    Detects pages containing excessive repeated lines.
    """
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if len(lines) < 10:
        return 0.0

    unique_lines = len(set(lines))

    return 1 - (unique_lines / len(lines))


def url_ratio(text):
    """
    Detect pages dominated by URLs.
    """
    if len(text) < 100:
        return 1.0

    urls = URL_PATTERN.findall(text)

    return sum(len(url) for url in urls) / len(text)


def looks_like_spam(text):
    lower = text.lower()

    matches = 0

    for pattern in SPAM_PATTERNS:
        if re.search(pattern, lower):
            matches += 1

    return matches >= 2


def quality_score(text):
    score = 100

    length = len(text)

    if length < MIN_CHARS:
        return 0

    if length > MAX_CHARS:
        score -= 20

    # Excessive repetition
    rep = repetition_ratio(text)

    if rep > 0.70:
        score -= 40
    elif rep > 0.50:
        score -= 20

    # URL-heavy content
    urls = url_ratio(text)

    if urls > 0.30:
        score -= 30
    elif urls > 0.15:
        score -= 15

    # Spam
    if looks_like_spam(text):
        score -= 50

    # Character diversity
    unique_chars = len(set(text))

    if unique_chars < 20:
        score -= 30

    return max(score, 0)


total = 0
kept = 0
removed = 0

print("Starting quality filtering...")

with open(INPUT, "r", encoding="utf-8") as fin, \
     open(OUTPUT, "w", encoding="utf-8") as fout:

    for line in fin:

        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        text = record.get("text", "")

        if not text:
            continue

        total += 1

        text = normalize_text(text)

        score = quality_score(text)

        if score >= 60:

            record["text"] = text
            record["quality_score"] = score

            fout.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

            kept += 1

        else:
            removed += 1

        if total % 500 == 0:
            print(
                f"Processed: {total:,} | "
                f"Kept: {kept:,} | "
                f"Removed: {removed:,}"
            )


print()
print("==============================")
print("QUALITY FILTER COMPLETE")
print("==============================")
print(f"Processed : {total:,}")
print(f"Kept      : {kept:,}")
print(f"Removed   : {removed:,}")
print(f"Output    : {OUTPUT}")