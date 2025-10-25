import argparse
from concurrent.futures import ThreadPoolExecutor

import litellm
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from extract.messages import get_all_messages
from features.user_features import get_user_features
from utils import file_cache
import config


MANAGER_MESSAGE_PROMPT = """
You are given a list of slack messages.

Your task is to:
- Examine extract the message to determine if it CONFIDENTLY indicates a manager-direct report relationship between two people.
- If it does, extract the relationship from the message.
- Explain the reason and give a confidence score between 0~1

In the response, only include the messages where you are able to extract a manager-direct report relationship.

List of messages:
{messages}
"""


class ManagerRelationship(BaseModel):
    message_id: str
    direct_report: str
    manager: str
    reason: str
    confidence: float


class ManagerRelationshipList(BaseModel):
    manager_relationships: list[ManagerRelationship]


@file_cache(f"{config.FEATURES_DATA_ROOT}/manager_messages.json")
def get_manager_messages(
    cutoff_date: pd.Timestamp = pd.to_datetime("2024-10-07"),
    max_message_length: int = 3000,
    manager_keywords: list[str] = ["manager", "reports to", "team lead"],
    batch_size: int = 100,
    max_workers: int = config.MAX_CONCURRENT_WORKERS,
) -> list[ManagerRelationship]:
    """
    Extract manager-relationship from messages.
    """
    # Load all users for ID -> name mapping
    user_features_df = get_user_features()

    # Get possible relevant management-related messages
    messages_df = get_all_messages()
    active_user_ids = set(user_features_df.index)
    mgmt_msgs = messages_df[
        # more recent dates
        (messages_df.timestamp >= cutoff_date)
        # manager keywords
        & (messages_df.text.apply(lambda t: any(keyword in t for keyword in manager_keywords)))
        # active users
        & (
            messages_df.user_id.isin(active_user_ids)
            | messages_df.mentions.apply(lambda m: len(set(m) & active_user_ids) > 0)
        )
        # message length
        & (messages_df.text.str.len() <= max_message_length)
    ]

    # Create message prompts
    msg_prompts = mgmt_msgs.apply(_get_message_prompt_str, axis=1).tolist()
    
    # Split into batches and process concurrently
    batches = [
        msg_prompts[i:i + batch_size]
        for i in range(0, len(msg_prompts), batch_size)
    ]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        batch_results = list(
            tqdm(
                executor.map(_process_message_batch, batches),
                total=len(batches),
                desc="Processing message batches",
                unit="batch",
            )
        )

    # Combine all results
    all_relationships = []
    for result in batch_results:
        all_relationships.extend(result.manager_relationships)
    return all_relationships


def _get_message_prompt_str(row: pd.Series) -> str:
    return f"""Message ID: {row["id"]}
Messenger: {row["user_name"]}
Message: {row["text_formatted"]}
"""


def _process_message_batch(batch_messages: list[str]) -> ManagerRelationshipList:
    """Process a batch of messages to extract manager relationships."""
    prompt = MANAGER_MESSAGE_PROMPT.format(messages="\n---\n".join(batch_messages))
    response = litellm.completion(
        model=config.DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=ManagerRelationshipList,
        metadata={"trace_name": "manager_messages"},
    )
    return ManagerRelationshipList.model_validate_json(response["choices"][0]["message"]["content"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract manager signals from access-requests"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force regeneration of cached signals",
    )
    args = parser.parse_args()

    get_manager_messages(force_refresh=args.force_refresh)
