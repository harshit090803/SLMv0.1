from warcio.archiveiterator import ArchiveIterator
import json

INPUT_FILE = "test.warc.wet"
OUTPUT_FILE = "test_extracted.jsonl"

count = 0
written = 0

with open(INPUT_FILE, "rb") as stream, open(
    OUTPUT_FILE, "w", encoding="utf-8"
) as output:

    for record in ArchiveIterator(stream):

        count += 1

        # WET files contain converted text records
        if record.rec_type != "conversion":
            continue

        try:
            text = record.content_stream().read().decode(
                "utf-8", errors="ignore"
            )

            url = record.rec_headers.get_header("WARC-Target-URI")

            if not text.strip():
                continue

            document = {
                "url": url,
                "text": text
            }

            output.write(
                json.dumps(
                    document,
                    ensure_ascii=False
                ) + "\n"
            )

            written += 1

        except Exception as e:
            print(f"Error processing record {count}: {e}")

        if written % 100 == 0:
            print(f"Extracted {written} documents...")

print("\n==============================")
print("WET extraction completed")
print("==============================")
print(f"Records scanned : {count}")
print(f"Documents saved : {written}")
print(f"Output          : {OUTPUT_FILE}")