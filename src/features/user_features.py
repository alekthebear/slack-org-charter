import pandas as pd

import config
from extract.users import get_users
from extract.messages import get_all_messages
from extract.channels import get_channels


def get_user_features() -> pd.DataFrame:
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
    recent_messages_df = messages_df[
        messages_df.timestamp >= config.RECENCY_CUT_OFF_DATE
    ]
    active_users_df = users_df[is_active & users_df.id.isin(recent_messages_df.user_id)]
    active_channels_df = channels_df[
        channels_df.id.isin(recent_messages_df.channel.unique())
    ]

    # Get basic aggregated statistics
    first_message_time_df = messages_df.groupby("user_id")["timestamp"].min()
    last_message_time_df = messages_df.groupby("user_id")["timestamp"].max()
    total_message_cnt = messages_df.groupby("user_id").size()
    recent_message_cnt = recent_messages_df.groupby("user_id").size()
    channels_created_cnt_df = channels_df.groupby("creator").size()
    channels_created_list_df = channels_df.groupby("creator")["name"].apply(list)
    recent_channels_created_cnt_df = (
        channels_df[channels_df.created >= config.RECENCY_CUT_OFF_DATE]
        .groupby("creator")
        .size()
    )
    recent_channels_created_list_df = (
        channels_df[channels_df.created >= config.RECENCY_CUT_OFF_DATE]
        .groupby("creator")["name"]
        .apply(list)
    )

    # Get mention counts
    recent_mention_count_df = (
        recent_messages_df.mentions.explode().dropna().value_counts()
    )
    total_mention_count_df = messages_df.mentions.explode().dropna().value_counts()

    # Get channel participation counts
    channels_participation_list = (
        messages_df.groupby(["user_id", "channel"])
        .size()
        .reset_index(name="cnt")  # More concise than .rename() then .reset_index()
        .sort_values(
            ["user_id", "cnt"], ascending=[True, False]
        )  # Sort before grouping
        .groupby("user_id")[["channel", "cnt"]]
        .apply(lambda x: x.to_dict(orient="records"))
    )
    channels_joined_cnt_df = channels_participation_list.apply(len)
    recent_channels_participation_list = (
        recent_messages_df.groupby(["user_id", "channel"])
        .size()
        .reset_index(name="cnt")
        .sort_values(["user_id", "cnt"], ascending=[True, False])
        .groupby("user_id")[["channel", "cnt"]]
        .apply(lambda x: x.to_dict(orient="records"))
    )
    recent_channels_joined_cnt_df = recent_channels_participation_list.apply(len)

    # user_info_df
    user_info_df = (
        active_users_df[["id", "full_name", "title"]]
        .set_index("id")
        .join(first_message_time_df.rename("first_message_time"))
        .join(last_message_time_df.rename("last_message_time"))
        .join(total_message_cnt.rename("total_message_count"))
        .join(recent_message_cnt.rename("recent_message_count"))
        .join(total_mention_count_df.rename("total_mention_count"))
        .join(recent_mention_count_df.rename("recent_mention_count"))
        .join(channels_created_cnt_df.rename("channels_created_count"))
        .join(channels_joined_cnt_df.rename("channels_joined_count"))
        .join(recent_channels_created_cnt_df.rename("recent_channels_created_count"))
        .join(recent_channels_joined_cnt_df.rename("recent_channels_joined_count"))
        .join(channels_created_list_df.rename("channels_created_list"))
        .join(recent_channels_created_list_df.rename("recent_channels_created_list"))
        .join(
            recent_channels_participation_list.rename(
                "recent_channels_participation_list"
            )
        )
        .join(channels_participation_list.rename("channels_participation_list"))
    )

    # Replace NaN values with default values
    count_columns = [
        "total_message_count",
        "recent_message_count",
        "total_mention_count",
        "recent_mention_count",
        "channels_created_count",
        "channels_joined_count",
        "recent_channels_joined_count",
        "recent_channels_created_count",
    ]
    user_info_df[count_columns] = user_info_df[count_columns].fillna(0).astype(int)
    list_columns = [
        "channels_created_list",
        "recent_channels_created_list",
        "recent_channels_participation_list",
        "channels_participation_list",
    ]
    for col in list_columns:
        user_info_df[col] = user_info_df[col].apply(
            lambda x: x if isinstance(x, list) else []
        )

    # Rate related features
    user_info_df["total_messages_per_day"] = user_info_df["total_message_count"] / (
        (messages_df.timestamp.max() - user_info_df["first_message_time"]).dt.days + 1
    )
    user_info_df["recent_messages_per_day"] = user_info_df["recent_message_count"] / (
        (messages_df.timestamp.max() - config.RECENCY_CUT_OFF_DATE).days + 1
    )
    user_info_df["total_mentions_per_day"] = user_info_df["total_mention_count"] / (
        (messages_df.timestamp.max() - user_info_df["first_message_time"]).dt.days + 1
    )
    user_info_df["recent_mentions_per_day"] = user_info_df["recent_mention_count"] / (
        (messages_df.timestamp.max() - config.RECENCY_CUT_OFF_DATE).days + 1
    )
    return user_info_df


if __name__ == "__main__":
    get_user_features()
