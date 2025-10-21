import argparse

import pandas as pd

import config
from extract.users import get_users
from extract.messages import get_all_messages
from extract.channels import get_channels
from utils import file_cache


@file_cache(f"{config.FEATURES_DATA_ROOT}/channel_features.parquet")
def get_channel_features() -> pd.DataFrame:
    users_df = get_users()
    messages_df = get_all_messages()
    channels_df = get_channels()

    # Determine active users
    is_active = (
        ~users_df["deleted"]
        & ~users_df["is_bot"]
        & ~users_df["is_restricted"]
        & ~users_df["is_ultra_restricted"]
    )
    active_users_df = users_df[
        is_active
        & users_df.id.isin(
            messages_df[messages_df.timestamp >= config.RECENCY_CUT_OFF_DATE].user_id
        )
    ]

    # We will only care about relevant messages from active users
    messages_df = messages_df[messages_df.user_id.isin(active_users_df.id)]
    recent_messages_df = messages_df[
        messages_df.timestamp >= config.RECENCY_CUT_OFF_DATE
    ]

    # Get basic aggregated statistics
    channel_message_cnt_df = messages_df.groupby("channel").size()
    channel_last_message_time_df = messages_df.groupby("channel")["timestamp"].max()
    recent_channel_message_cnt_df = recent_messages_df.groupby("channel").size()

    # channel_info_df
    channel_info_df = (
        channels_df.set_index("name")
        .join(channel_message_cnt_df.rename("total_message_count"))
        .join(recent_channel_message_cnt_df.rename("recent_message_count"))
        .join(channel_last_message_time_df.rename("last_message_time"))
        .reset_index()
    )

    # Replace NaN values with default values
    count_columns = [
        "total_message_count",
        "recent_message_count",
    ]
    channel_info_df[count_columns] = (
        channel_info_df[count_columns].fillna(0).astype(int)
    )

    # Other channel features
    channel_info_df["total_messages_per_day"] = channel_info_df[
        "total_message_count"
    ] / ((messages_df.timestamp.max() - channel_info_df["created"]).dt.days + 1)
    channel_info_df["recent_messages_per_day"] = channel_info_df[
        "recent_message_count"
    ] / ((messages_df.timestamp.max() - config.RECENCY_CUT_OFF_DATE).days + 1)
    channel_info_df["age"] = (
        channel_info_df["last_message_time"] - channel_info_df["created"]
    )
    channel_info_df["is_active"] = (
        channel_info_df["name"].isin(recent_messages_df.channel.unique()).astype(bool)
    )
    return channel_info_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate channel features")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force regeneration of cached features",
    )
    args = parser.parse_args()

    get_channel_features(force_refresh=args.force_refresh)
