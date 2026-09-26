"""Application configuration and constants.

Centralising these values makes slot definitions, capacities and secrets
easy to find and change, instead of being scattered as magic literals
through the application code.
"""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent

# Time slots offered by the procurement centre.
SLOTS = [
    "08:00-09:00", "09:00-10:00", "10:00-11:00", "11:00-12:00",
    "12:00-13:00", "14:00-15:00", "15:00-16:00", "16:00-17:00",
]

# Peak slots (index-based) get a larger capacity allowance.
_PEAK_SLOT_INDEXES = {0, 1, 4, 7}
SLOT_CAPACITY = {
    slot: (20 if i in _PEAK_SLOT_INDEXES else 25)
    for i, slot in enumerate(SLOTS)
}


class Config:
    """Flask configuration, overridable via environment variables."""

    SECRET_KEY = os.environ.get("SECRET_KEY", "local-academic-demo")
    DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    DATABASE_PATH = ROOT_DIR / "procurement.db"
    MODELS_DIR = ROOT_DIR / "models"
    DATASET_PATH = ROOT_DIR / "procurement_data.csv"

    # Demo credentials for the login gate. Override via environment
    # variables for anything beyond an academic demo.
    LOGIN_USERNAME = os.environ.get("LOGIN_USERNAME", "admin")
    LOGIN_PASSWORD = os.environ.get("LOGIN_PASSWORD", "admin123")
