import argparse

import pandas as pd

import config
from features.manager_messages import get_manager_messages
from features.user_features import get_user_features
from utils import file_cache, get_timestamp_from_id


@file_cache(f"{config.FEATURES_DATA_ROOT}/explicit_managers.txt")
def get_explicit_managers(
    confidence_threshold: float = 0.8,
) -> str:
    """
    Process manager messages to extract manager relationships.
    """
    user_features_df = get_user_features()
    mgmt_relationships = get_manager_messages()

    # Filter by confidence
    mgmt_relationships = [
        r
        for r in mgmt_relationships
        if (
            r.confidence >= confidence_threshold
            and r.direct_report in set(user_features_df.full_name)
            and r.manager in set(user_features_df.full_name)
        )
    ]

    mgmt_df = pd.DataFrame.from_dict([r.model_dump() for r in mgmt_relationships])
    mgmt_df["timestamp"] = mgmt_df["message_id"].map(get_timestamp_from_id)
    mgmt_df = (
        mgmt_df.sort_values("timestamp", ascending=False)
        .groupby("direct_report")
        .first()
        .reset_index()
    )
    return _get_explicit_manager_str(mgmt_df)


def _get_explicit_manager_str(mgmt_df: pd.DataFrame) -> str:
    return "\n".join(
        [
            f"- {row['direct_report']} reports to {row['manager']} (message date: {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')})"
            for _, row in mgmt_df.iterrows()
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get the latest high confidence manager messages"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force regeneration of cached features",
    )
    args = parser.parse_args()
    mgmt_df = get_explicit_managers()
