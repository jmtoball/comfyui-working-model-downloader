"""Credentials: where they come from, and where they must never end up."""

from __future__ import annotations

import json
import os
import stat

from wmd import config


def test_the_environment_supplies_tokens_by_default(folder_paths, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    monkeypatch.setenv("CIVITAI_API_KEY", "civitai_from_env")
    cfg = config.load()
    assert cfg.hf_token == "hf_from_env" and cfg.hf_token_source == "env"
    assert cfg.civitai_api_key == "civitai_from_env"


def test_the_huggingface_cli_token_file_is_honoured(folder_paths, tmp_path, monkeypatch):
    home = tmp_path / "hf"
    home.mkdir()
    (home / "token").write_text("hf_from_cli\n")
    monkeypatch.setenv("HF_HOME", str(home))
    cfg = config.load()
    assert cfg.hf_token == "hf_from_cli" and cfg.hf_token_source == "hf-cli"


def test_a_value_set_in_the_panel_overrides_the_environment(folder_paths, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    cfg = config.save({"hf_token": "hf_from_panel"})
    assert cfg.hf_token == "hf_from_panel" and cfg.hf_token_source == "user"


def test_clearing_a_panel_value_falls_back_to_the_environment(folder_paths, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    config.save({"hf_token": "hf_from_panel"})
    cfg = config.save({"hf_token": ""})
    assert cfg.hf_token == "hf_from_env" and cfg.hf_token_source == "env"


def test_the_config_file_lives_in_the_user_directory_not_the_extension(folder_paths):
    config.save({"hf_token": "x"})
    assert config.config_path().startswith(folder_paths.root)
    assert "custom_nodes" not in config.config_path()


def test_the_config_file_is_not_world_readable(folder_paths):
    config.save({"hf_token": "secret"})
    mode = stat.S_IMODE(os.stat(config.config_path()).st_mode)
    assert mode & (stat.S_IRGRP | stat.S_IROTH) == 0


def test_the_panel_only_ever_sees_a_masked_token(folder_paths):
    config.save({"hf_token": "hf_abcdefghijklmnop"})
    view = config.masked()
    assert view["hf_token_set"] is True
    assert view["hf_token_hint"] == "…mnop"
    assert "hf_abcdefghijklmnop" not in json.dumps(view)


def test_secrets_are_scrubbed_from_anything_we_might_print(folder_paths):
    cfg = config.save({"civitai_api_key": "civitai_secret_value"})
    message = "failed: https://civitai.com/x?token=civitai_secret_value"
    assert "civitai_secret_value" not in config.redact(message, cfg)


def test_concurrency_is_clamped_to_something_sane(folder_paths):
    assert config.save({"prefs": {"max_concurrent_downloads": 99}}).prefs.max_concurrent_downloads == 8
    assert config.save({"prefs": {"max_concurrent_downloads": 0}}).prefs.max_concurrent_downloads == 1


def test_a_corrupt_config_file_does_not_break_startup(folder_paths):
    os.makedirs(os.path.dirname(config.config_path()), exist_ok=True)
    with open(config.config_path(), "w", encoding="utf-8") as handle:
        handle.write("{ not json")
    assert config.load().hf_token == ""
