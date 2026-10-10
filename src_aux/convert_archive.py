import argparse
import json
import os
import random

ARCHIVE_DIR = os.path.join("data", "archive", "converted_apc")


def convert_split(src, dst, category_field, fraction=1.0, seed=42):
    """Writes one `text####[[aspect, category, polarity, opinion], ...]` line per JSONL row; returns the categories seen."""
    categories = set()
    with open(src, encoding="utf-8") as fin:
        rows = [json.loads(line) for line in fin]
    if fraction < 1.0:
        keep = sorted(random.Random(seed).sample(range(len(rows)), round(len(rows) * fraction)))
        rows = [rows[i] for i in keep]
    with open(dst, "w", encoding="utf-8") as fout:
        for row in rows:
            text = " ".join(row["text"].split())
            quads = []
            for q in row["quads"]:
                category = q[category_field] or "NULL"
                categories.add(category)
                # 'NULL' marks implicit aspects/opinions, as in the other data/raw datasets
                quads.append([q["aspect_term"] or "NULL", category, q["sentiment"], q["opinion_term"] or "NULL"])
            fout.write(f"{text}####{quads!r}\n")
    return categories, len(rows)


def write_categories_toml(categories, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("categories = [\n")
        for category in sorted(categories):
            f.write(f"    {json.dumps(category)},\n")
        f.write("]\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert data/archive JSONL into the '####' format used in data/raw.")
    parser.add_argument("--name", default="hotels_vn", help="dataset name used under data/raw/ and config/")
    parser.add_argument("--category-field", default="aspect_category", choices=["aspect_category", "taxonomy_code"],
                        help="coarse (6 classes) or fine-grained (31 classes) category label")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="keep only this share of every split (seeded random sample), e.g. 0.1 for a demo set")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = os.path.join("data", "raw", args.name)
    os.makedirs(out_dir, exist_ok=True)
    train_categories = None
    for split in ("train", "dev", "test"):
        categories, n_rows = convert_split(os.path.join(ARCHIVE_DIR, f"{split}.jsonl"),
                                           os.path.join(out_dir, f"{split}.txt"), args.category_field,
                                           args.fraction, args.seed)
        if split == "train":
            train_categories = categories
        print(f"{split}: {n_rows} rows written to {os.path.join(out_dir, split + '.txt')}")

    # Same convention as src_aux/extract_categories.py: the label list comes from the training split.
    toml_path = os.path.join("config", f"{args.name}_categories.toml")
    write_categories_toml(train_categories, toml_path)
    print(f"{len(train_categories)} categories written to {toml_path}")
