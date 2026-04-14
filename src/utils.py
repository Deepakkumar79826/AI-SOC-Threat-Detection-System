import json
from pathlib import Path

def save_json(data: dict, filepath: Path) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)