import os
from pathlib import Path

import pandas as pd


# ====================
#  File and Data Paths
# ====================
RAW_DATA_ROOT = os.environ.get("RAW_DATA_ROOT", f"{Path.home()}/Desktop/OrgChartTakehomeOctober/data/raw")
CACHE_ROOT = os.environ.get("CACHE_ROOT", f"{Path.home()}/Desktop/OrgChartTakehomeOctober/cache")

# (Derived from root directories specified above)
EXTRACTED_DATA_ROOT = f"{CACHE_ROOT}/extracted"
FEATURES_DATA_ROOT = f"{CACHE_ROOT}/features"
INFERENCE_DATA_ROOT = f"{CACHE_ROOT}/inference"
USERS_FILE_PATH = f"{RAW_DATA_ROOT}/users.json"
CHANNELS_FILE_PATH = f"{RAW_DATA_ROOT}/channels.json"


# ====================
#  Default Parameters
# ====================
COMPANY_NAME = "Gather Town"
RECENCY_CUT_OFF_DATE = pd.to_datetime("2025-09-01")
MAX_CONCURRENT_WORKERS = 10
DEFAULT_MODEL = "openai/gpt-5"
WEB_SEARCH_MODEL = "openai/gpt-5-search-api"