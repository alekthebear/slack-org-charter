from functools import cache
import json

import pandas as pd

import config


def pretty_print_messages(
    messages_df: pd.DataFrame, return_string: bool = False
) -> str | None:
    """For debugging/development purposes, print the messages in a readable format."""
    messages_df = messages_df.sort_values(by=["channel", "timestamp"], ascending=True)
    result = ""
    for channel, channel_messages_df in messages_df.groupby("channel"):
        result += "--------------------------------\n"
        result += f"Channel: {channel}\n"
        result += f"Message Count: {len(channel_messages_df)}\n"
        result += f"Message Range: {channel_messages_df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')} to {channel_messages_df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')}\n"
        result += "--------------------------------\n"
        for _, row in channel_messages_df.iterrows():
            timestamp = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            user_name = row["user_name"]
            text_formatted = row["text_formatted"]
            result += f"({timestamp}) {user_name}: {text_formatted}\n"
        result += "\n"
    if return_string:
        return result
    else:
        print(result)


@cache
def get_user_id_to_name_map(user_file_path: str = config.USERS_FILE_PATH) -> dict:
    with open(user_file_path) as f:
        users = json.load(f)
    return {u["id"]: u["profile"]["real_name_normalized"] for u in users}
