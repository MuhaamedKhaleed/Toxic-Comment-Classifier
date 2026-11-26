"""Explicit dotenv configuration. No secrets are required."""

import os
from pathlib import Path

from dotenv import load_dotenv


def local_model_path(explicit: Path | None, env_file: Path | None = None) -> Path:
    """Prefer CLI path, then TOXIC_MODEL_DIR from process or explicit dotenv."""
    if env_file is not None:
        if not env_file.is_file():
            raise ValueError("Env file not found. Supply an existing --env-file or omit it.")
        load_dotenv(env_file, override=False)
    value = str(explicit) if explicit is not None else os.getenv("TOXIC_MODEL_DIR", "")
    if not value.strip():
        raise ValueError("Provide --model-dir or set TOXIC_MODEL_DIR in an explicit dotenv file.")
    return Path(value).expanduser()
