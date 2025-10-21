import json

import pandas as pd

import config


def get_channels(channel_file_path: str = config.CHANNELS_FILE_PATH) -> pd.DataFrame:
    with open(channel_file_path) as f:
        channels = json.load(f)
    return pd.DataFrame.from_records([extract_channel(c) for c in channels])


def extract_channel(channel_dict: dict) -> dict:
    return {
        "id": channel_dict["id"],
        "name": channel_dict["name"],
        "created": pd.to_datetime(channel_dict["created"], unit="s"),
        "creator": channel_dict["creator"],
        "is_archived": channel_dict["is_archived"],
        "is_general": channel_dict["is_general"],
        "members": channel_dict["members"],
        "topic": channel_dict["topic"]["value"],
        "topic_creator": channel_dict["topic"]["creator"],
        "purpose": channel_dict["purpose"]["value"],
        "purpose_creator": channel_dict["purpose"]["creator"],
    }
