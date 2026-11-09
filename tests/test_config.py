from pathlib import Path

import pytest
import yaml

from src.config import load_items, require_api_key


def test_load_items_outside_project(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config = load_items()
    assert len(config["items"]) == 5
    assert config["items"]["배추"]["hsk"] == ["0704902000"]
    assert config["items"]["양파"]["stations"][0]["months"] == "all"
    assert config["ecos"]["items"]["usd"] == "0000001"


@pytest.mark.parametrize("content", ["", "[]", "items: {}", "items: []"])
def test_invalid_structure(tmp_path, content):
    path = tmp_path / "items.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_items(path)


@pytest.mark.parametrize("field,value", [("type", "invalid"), ("hsk", [704902000]), ("hsk", [])])
def test_invalid_item(tmp_path, field, value):
    config = load_items()
    config["items"]["배추"][field] = value
    path = tmp_path / "items.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        load_items(path)


def test_missing_item_setting(tmp_path):
    config = load_items()
    del config["items"]["배추"]["stations"]
    path = tmp_path / "items.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="stations"):
        load_items(path)


@pytest.mark.parametrize("value", [None, "", "  "])
def test_missing_api_key(monkeypatch, value):
    monkeypatch.delenv("ECOS_KEY", raising=False)
    if value is not None:
        monkeypatch.setenv("ECOS_KEY", value)
    with pytest.raises(ValueError, match="ECOS_KEY"):
        require_api_key("ECOS_KEY")


def test_api_key_from_environment(monkeypatch):
    monkeypatch.setenv("ECOS_KEY", "test-only-value")
    assert require_api_key("ECOS_KEY") == "test-only-value"


def test_unsupported_api_key():
    with pytest.raises(ValueError, match="Unsupported"):
        require_api_key("UNKNOWN_KEY")


def test_loader_does_not_read_dotenv(monkeypatch):
    original = Path.open

    def guarded_open(path, *args, **kwargs):
        assert not path.name.startswith(".env")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    load_items()
