import argparse
import json


def convert_jsonl_to_json(input_path, output_path):
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL to JSON")
    parser.add_argument("input", help="Input .jsonl file path")
    parser.add_argument("output", help="Output .json file path")
    args = parser.parse_args()

    convert_jsonl_to_json(args.input, args.output)
    print(f"Done: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
