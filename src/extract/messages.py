import json
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm.auto import tqdm

import consts
from utils import get_first_value


def extract_all_messages(
    data_root: str = consts.RAW_DATA_ROOT,
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Extract all messages from all channel directories in the data root.

    Args:
        data_root: Root directory containing channel subdirectories
        cache_dir: Directory to save/load cached parquet files. Defaults to data_root/processed
        use_cache: If True, load from cache if available and save to cache after extraction

    Returns a DataFrame with one row per message containing:
    - message metadata (user, timestamp, type, etc.)
    - extracted signals (mentions, thread info, reactions)
    - channel context
    """
    cache_dir = cache_dir or consts.PROCESSED_DATA_ROOT
    all_messages = []

    # Iterate through all channel directories
    for channel_dir in tqdm(list(Path(data_root).glob("*/")), desc="Channels"):
        channel_name = channel_dir.name
        channel_messages_df = extract_channel_messages(
            channel_name,
            data_root=data_root,
            cache_dir= cache_dir if use_cache else None,
        )
        all_messages.append(channel_messages_df)

    if all_messages:
        return pd.concat(all_messages, ignore_index=True)
    else:
        return pd.DataFrame()


def extract_channel_messages(
    channel_name: str,
    data_root: str = consts.RAW_DATA_ROOT,
    cache_dir: Optional[str] = consts.PROCESSED_DATA_ROOT,
) -> pd.DataFrame:
    """
    Extract messages from a specific channel.

    Args:
        channel_name: Name of the channel directory
        data_root: Root directory containing channel subdirectories
        cache_dir: Optional directory to save cached parquet file.
                   If provided and file doesn't exist, saves after extraction.

    Returns:
        DataFrame with extracted message data
    """
    messages = []
    channel_path = Path(data_root) / channel_name
    if not channel_path.exists():
        raise ValueError(f"Channel directory not found: {channel_path}")

    for json_file in sorted(channel_path.glob("*.json")):
        try:
            with open(json_file) as f:
                daily_messages = json.load(f)

            for msg in daily_messages:
                extracted = extract_message(msg, channel_name)
                if extracted:
                    messages.append(extracted)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing {json_file}: {e}")
            continue

    df = pd.DataFrame.from_records(messages)

    # Save to cache if filepath specified
    cache_file_path = Path(cache_dir) / f"{channel_name}.parquet" if cache_dir else None
    if cache_file_path and not df.empty:
        df.to_parquet(cache_file_path, index=False)
        print(f"Saved {channel_name} to cache: {cache_file_path}")
    return df


def extract_message(msg: dict, channel_name: str) -> Optional[dict]:
    """
    Extract relevant fields from a single message.
    """
    # filter out non-message events
    if "type" not in msg or msg["type"] != "message":
        return None

    # User info
    user_id = msg.get("user")

    # Extract timestamp
    ts = msg["ts"]
    timestamp = pd.to_datetime(float(ts), unit="s")

    # Extract text content
    text = msg.get("text", "")

    # Extract mentions from text and blocks
    mentions = _extract_mentions(msg)

    # Extract thread information
    thread_ts = msg.get("thread_ts")
    parent_thread_id = _get_message_id(channel_name, thread_ts) if thread_ts else None
    parent_user_id = msg.get("parent_user_id")
    reply_users = msg.get("reply_users", [])
    reply_count = msg.get("reply_count", 0)
    is_thread_parent = thread_ts == ts if thread_ts and ts else False

    # Extract reactions
    reactions = _extract_reactions(msg)

    # Extract files (indicates content creation)
    has_files = len(msg.get("files", [])) > 0
    file_count = len(msg.get("files", []))

    return {
        # Unique Identifier
        "id": _get_message_id(channel_name, ts),
        # User info
        "user_id": user_id,
        # Message metadata
        "timestamp": timestamp,
        "channel": channel_name,
        "text": text,
        "message_type": msg.get("subtype", "normal"),
        # Interaction signals
        "mentions": mentions,  # List of user_ids mentioned
        "mention_count": len(mentions),
        # Thread signals
        "parent_thread_id": parent_thread_id,
        "parent_user_id": parent_user_id,
        "is_in_thread": bool(thread_ts),
        "is_thread_parent": is_thread_parent,
        "reply_users": reply_users,
        "reply_count": reply_count,
        # Reaction signals
        "reactions": reactions,  # List of {name, users, count}
        "reaction_count": sum(r["count"] for r in reactions),
        "reaction_users": list(set([u for r in reactions for u in r["users"]])),
        # Content signals
        "has_files": has_files,
        "file_count": file_count,
    }


def _extract_mentions(msg: dict) -> list[str]:
    """
    Extract all user mentions from a message.

    Looks in both the text field and the structured blocks.
    Returns a list of unique user IDs.
    """
    mentions = set()

    # Extract from text field (format: <@USER_ID>)
    text = msg.get("text", "")
    import re

    text_mentions = re.findall(r"<@([A-Z0-9]+)>", text)
    mentions.update(text_mentions)

    # Extract from blocks (more reliable for rich text)
    blocks = msg.get("blocks", [])
    for block in blocks:
        if block.get("type") == "rich_text":
            for element in block.get("elements", []):
                if element.get("type") == "rich_text_section":
                    for item in element.get("elements", []):
                        if item.get("type") == "user":
                            user_id = item.get("user_id")
                            if user_id:
                                mentions.add(user_id)

    return list(mentions)


def _extract_reactions(msg: dict) -> list[dict]:
    """
    Extract reaction information from a message.

    Returns a list of dicts with {name, users, count}
    """
    reactions = msg.get("reactions", [])
    return [
        {"name": r["name"], "users": r["users"], "count": r["count"]} for r in reactions
    ]

def _get_message_id(channel_name: str, ts: str) -> str:
    """
    Generate a unique message ID based on channel name and timestamp.
    """
    return f"{channel_name}_{ts}"