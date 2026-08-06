import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

AUTO_JSON_PATH = object()


def load_meta_info(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Meta info file not found: {path}")

    with path.open("rb") as file:
        meta_info = pickle.load(file)

    if not isinstance(meta_info, dict):
        raise TypeError(
            f"Unexpected meta info format: {type(meta_info)}, expected dict."
        )
    return meta_info


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(key): to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(value) for value in obj]
    return obj


def describe_array(name: str, array: np.ndarray) -> str:
    summary = f"{name}: shape={array.shape}, dtype={array.dtype}"
    if array.size == 0:
        return f"{summary}, empty"

    flattened = array.ravel()
    if np.issubdtype(array.dtype, np.floating):
        preview = ", ".join(f"{value:.4f}" for value in flattened[:6])
    else:
        preview = ", ".join(str(value) for value in flattened[:6])
    if flattened.size > 6:
        preview += ", ..."
    return f"{summary}, preview=[{preview}]"


def print_section(title: str, data: Any, indent: int = 0) -> None:
    prefix = "  " * indent
    if isinstance(data, dict):
        print(f"{prefix}{title}:")
        for key, value in data.items():
            print_section(str(key), value, indent + 1)
    elif isinstance(data, np.ndarray):
        print(f"{prefix}{describe_array(title, data)}")
    elif isinstance(data, (list, tuple)):
        print(f"{prefix}{title}: list(len={len(data)})")
        for index, item in enumerate(data[:5]):
            print_section(f"[{index}]", item, indent + 1)
        if len(data) > 5:
            print(f"{prefix}  ... ({len(data) - 5} more items)")
    else:
        print(f"{prefix}{title}: {data}")


def print_summary(meta_info: dict[str, Any]) -> None:
    state = meta_info.get("state", {})
    action = meta_info.get("action", {})
    data = meta_info.get("data", {})
    image = meta_info.get("image", {})
    policy = meta_info.get("policy", {})

    print("=== Model Meta Info Summary ===")

    if state:
        print("\n[State]")
        print(f"  keys: {state.get('keys')}")
        for key in ("example", "mean", "std"):
            value = state.get(key)
            if isinstance(value, np.ndarray):
                print(f"  {describe_array(key, value)}")

    if action:
        print("\n[Action]")
        print(f"  keys: {action.get('keys')}")
        for key in ("example", "mean", "std"):
            value = action.get(key)
            if isinstance(value, np.ndarray):
                print(f"  {describe_array(key, value)}")

    if image:
        print("\n[Image]")
        for key, value in image.items():
            print_section(str(key), value, indent=1)

    if data:
        print("\n[Data]")
        for key, value in data.items():
            print_section(str(key), value, indent=1)

    if policy:
        print("\n[Policy]")
        for key, value in policy.items():
            print_section(str(key), value, indent=1)

    standard_keys = {"state", "action", "image", "data", "policy"}
    remaining = {
        key: value for key, value in meta_info.items() if key not in standard_keys
    }
    if remaining:
        print("\n[Other]")
        for key, value in remaining.items():
            print_section(str(key), value, indent=1)

    print("\n=== End Summary ===")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display the contents of a model_meta_info.pkl file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("path", type=Path, help="path to model_meta_info.pkl")
    parser.add_argument(
        "--save-json",
        type=Path,
        nargs="?",
        const=AUTO_JSON_PATH,
        default=None,
        metavar="PATH",
        help=(
            "save the meta info as JSON; when PATH is omitted, save it next to "
            "the input pkl file"
        ),
    )
    parser.add_argument(
        "--full",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="print the full nested structure instead of a summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta_info = load_meta_info(args.path)
    print(f"Loaded meta info from: {args.path.resolve()}")

    if args.full:
        print_section("meta_info", meta_info)
    else:
        print_summary(meta_info)

    if args.save_json is not None:
        json_path = (
            args.path.with_suffix(".json")
            if args.save_json is AUTO_JSON_PATH
            else args.save_json
        )
        serializable = to_serializable(meta_info)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as file:
            json.dump(serializable, file, ensure_ascii=False, indent=2)
        print(f"\nSaved JSON representation to: {json_path.resolve()}")


if __name__ == "__main__":
    main()
