import json

import pandas as pd

import consts


USER_FILE_PATH = consts.RAW_DATA_ROOT + "/" + "users.json"


def extract_users(user_file_path: str = USER_FILE_PATH) -> pd.DataFrame:
    with open(user_file_path) as f:
        users = json.load(f)
    return pd.DataFrame.from_records([extract_user(u) for u in users])


def extract_user(user_dict: dict) -> dict:
    return {
        "id": user_dict["id"],
        "name": user_dict["name"],
        "deleted": user_dict["deleted"],
        "title": user_dict["profile"]["title"],
        "full_name": user_dict["profile"]["real_name_normalized"],
        "display_name": user_dict["profile"]["display_name_normalized"],
        "is_bot": user_dict["is_bot"],
        "is_restricted": user_dict.get("is_restricted", False),
        "is_ultra_restricted": user_dict.get("is_ultra_restricted", False),
        "updated": pd.to_datetime(user_dict["updated"], unit="s"),
    }
