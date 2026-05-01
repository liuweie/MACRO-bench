import argparse
import json

import yaml


def convert_yaml_to_json(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert YAML file to JSON file")
    parser.add_argument("input", help="Input YAML file path")
    parser.add_argument("output", help="Output JSON file path")
    args = parser.parse_args()

    convert_yaml_to_json(args.input, args.output)
    print(f"Done: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
