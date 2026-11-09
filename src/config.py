import os
from pathlib import Path

import yaml


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
API_KEY_NAMES = frozenset({
    "DATA_GO_KR_KEY", "KAMIS_KEY", "KAMIS_ID", "ECOS_KEY",
    "GARAK_ID", "GARAK_PASSWD",
})


def load_items(path: str | Path = CONFIG_DIR / "items.yaml") -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("items"), dict):
        raise ValueError("Item configuration must contain an items mapping")
    expected = {"배추", "무", "양파", "깐마늘", "건고추"}
    if set(config["items"]) != expected:
        raise ValueError("Item configuration must contain exactly the five project items")
    for name, item in config["items"].items():
        if not isinstance(item, dict):
            raise ValueError(f"Invalid item configuration: {name}")
        required = {"type", "match", "kamis_csv", "hsk", "stations"}
        missing = required - item.keys()
        if missing:
            raise ValueError(f"Missing settings for {name}: {', '.join(sorted(missing))}")
        if item["type"] not in {"fresh", "storage"}:
            raise ValueError(f"Invalid item type: {name}")
        if not isinstance(item["hsk"], list) or not item["hsk"] or any(
            not isinstance(code, str) or len(code) != 10 or not code.isascii() or not code.isdigit()
            for code in item["hsk"]
        ):
            raise ValueError(f"HSK codes must be 10-digit strings: {name}")
    return config


def require_api_key(name: str) -> str:
    if name not in API_KEY_NAMES:
        raise ValueError("Unsupported API key variable")
    if not os.environ.get(name, "").strip():
        raise ValueError(f"Missing required environment variable: {name}")
    return os.environ[name]
