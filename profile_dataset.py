import json
import re
from collections import Counter

INPUT_FILE = "test_extracted.jsonl"

total = 0
empty = 0
short = 0
long = 0
characters = 0

languages = Counter()

def rough_language(text):
    """
    Very rough script-based language detection.
    This is only for profiling, not final training.
    """

    devanagari = len(re.findall(r'[\u0900-\u097F]', text))
    bengali = len(re.findall(r'[\u0980-\u09FF]', text))
    gurmukhi = len(re.findall(r'[\u0A00-\u0A7F]', text))
    gujarati = len(re.findall(r'[\u0A80-\u0AFF]', text))
    odia = len(re.findall(r'[\u0B00-\u0B7F]', text))
    tamil = len(re.findall(r'[\u0B80-\u0BFF]', text))
    telugu = len(re.findall(r'[\u0C00-\u0C7F]', text))
    kannada = len(re.findall(r'[\u0C80-\u0CFF]', text))
    malayalam = len(re.findall(r'[\u0D00-\u0D7F]', text))
    latin = len(re.findall(r'[A-Za-z]', text))

    counts = {
        "Devanagari": devanagari,
        "Bengali": bengali,
        "Gurmukhi": gurmukhi,
        "Gujarati": gujarati,
        "Odia": odia,
        "Tamil": tamil,
        "Telugu": telugu,
        "Kannada": kannada,
        "Malayalam": malayalam,
        "Latin": latin,
    }

    best = max(counts, key=counts.get)

    if counts[best] < 20:
        return "Unknown"

    return best


with open(INPUT_FILE, "r", encoding="utf-8") as f:

    for line in f:

        try:
            doc = json.loads(line)

            text = doc.get("text", "").strip()

            total += 1
            length = len(text)
            characters += length

            if not text:
                empty += 1

            if length < 200:
                short += 1

            if length > 100000:
                long += 1

            lang = rough_language(text)
            languages[lang] += 1

        except Exception:
            continue


print("\n==============================")
print("INDAI DATASET PROFILE")
print("==============================")

print(f"Documents       : {total:,}")
print(f"Empty documents : {empty:,}")
print(f"Short (<200)    : {short:,}")
print(f"Long (>100k)    : {long:,}")
print(f"Total characters: {characters:,}")

if total:
    print(f"Average length  : {characters / total:,.0f}")

print("\nLanguage/script distribution:")
print("------------------------------")

for language, count in languages.most_common():
    percentage = count / total * 100
    print(f"{language:15} {count:8,} ({percentage:5.2f}%)")