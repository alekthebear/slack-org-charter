import argparse


import config
from extract.messages import get_all_messages
from extract.users import get_users
from utils import file_cache, get_user_id_to_name_map


@file_cache(f"{config.FEATURES_DATA_ROOT}/user_mention_graph.json")
def get_user_mention_graph() -> dict:
    """
    Compute pairwise mention relationships from recent Slack messages.

    Returns a nested dictionary mapping:
        {user_id: {mentioned_user_id: count}}

    For example:
        {"U123": {"U456": 10, "U789": 5}}
    means user U123 mentioned U456 10 times and U789 5 times in recent messages.
    """
    user_id_to_name_map = get_user_id_to_name_map()
    messages_df = get_all_messages()
    users_df = get_users()

    # Filter for recent messages only
    recent_messages_df = messages_df[
        messages_df.timestamp >= config.RECENCY_CUT_OFF_DATE
    ]

    # Filter for active users (same logic as user_features.py)
    is_active = (
        ~users_df["deleted"]
        & ~users_df["is_bot"]
        & ~users_df["is_restricted"]
        & ~users_df["is_ultra_restricted"]
    )
    active_user_ids = set(users_df[is_active]["id"])

    # Build mention graph
    recent_mentions = (
        recent_messages_df[["user_id", "user_name", "mentions", "channel"]]
        .explode("mentions")
        .dropna()
    )
    recent_mentions = recent_mentions[
        recent_mentions.user_id.isin(active_user_ids)
        & recent_mentions.mentions.isin(active_user_ids)
    ]
    recent_mentions["user_name"] = recent_mentions.user_id.apply(lambda x: user_id_to_name_map[x])
    recent_mentions["mentions"] = recent_mentions.mentions.apply(lambda x: user_id_to_name_map[x])
    mention_graph = recent_mentions.groupby(["user_name", "mentions"]).size().reset_index(name="count")
    return mention_graph.to_dict(orient="records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate user mention graph from recent messages"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force regeneration of cached mention graph",
    )
    args = parser.parse_args()

    mention_graph = get_user_mention_graph(force_refresh=args.force_refresh)
