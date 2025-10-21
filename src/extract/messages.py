from pathlib import Path
from typing import Optional
import json
import re

from tqdm.auto import tqdm
import pandas as pd

import config
import utils


def get_all_messages(
    data_root: str = config.RAW_DATA_ROOT,
    cache_root: str = config.EXTRACTED_DATA_ROOT,
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
    all_messages = []

    # Iterate through all channel directories
    for channel_dir in tqdm(
        list(Path(data_root).glob("*/")), desc="Load Channel Messages"
    ):
        channel_name = channel_dir.name
        cache_path = Path(cache_root) / f"{channel_name}.parquet"
        if use_cache and cache_path and cache_path.exists():
            channel_messages_df = pd.read_parquet(cache_path)
        else:
            channel_messages_df = get_channel_messages(
                channel_name,
                data_root=data_root,
                cache_root=cache_root,
                use_cache=use_cache,
            )
        all_messages.append(channel_messages_df)

    if all_messages:
        return pd.concat(all_messages, ignore_index=True)
    else:
        return pd.DataFrame()


def get_channel_messages(
    channel_name: str,
    data_root: str = config.RAW_DATA_ROOT,
    cache_root: str = config.EXTRACTED_DATA_ROOT,
    use_cache: bool = True,
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
    cache_path = Path(cache_root) / f"{channel_name}.parquet"
    if use_cache and cache_path and cache_path.exists():
        print(f"Loaded {channel_name} from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    messages = []
    input_path = Path(data_root) / channel_name
    if not input_path.exists():
        raise ValueError(f"Channel directory not found: {input_path}")
    for json_file in sorted(input_path.glob("*.json")):
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

    messages_df = pd.DataFrame.from_records(messages)

    # Save to cache if filepath specified
    if not messages_df.empty:
        messages_df.to_parquet(cache_path, index=False)
        print(f"Saved {channel_name} messagesto cache: {cache_path}")
    return messages_df


def extract_message(msg: dict, channel_name: str) -> Optional[dict]:
    """
    Extract relevant fields from a single message.
    """
    user_id_to_name_map = utils.get_user_id_to_name_map()

    # filter out non-message events
    if "type" not in msg or msg["type"] != "message":
        return None

    # User info
    user_id = msg.get("user")
    user_name = user_id_to_name_map.get(user_id, "")

    # Extract timestamp
    ts = msg["ts"]
    timestamp = pd.to_datetime(float(ts), unit="s")

    # Extract text content
    text = msg.get("text", "")

    # Create text with mentions replaced by names
    text_formatted = _format_names(text, user_id_to_name_map)

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
        "user_name": user_name,
        # Message metadata
        "timestamp": timestamp,
        "channel": channel_name,
        "text": text,
        "text_formatted": text_formatted,  # Text with mentions replaced by names
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


def _format_names(text: str, user_id_to_name_map: dict) -> str:
    """
    Replace user ID mentions in text with user names. e.g. <@U018H1KULD8> -> <@First Last>

    Args:
        text: Message text containing mentions in format <@USER_ID>
        user_id_to_name_map: Dict mapping user IDs to names

    Returns:
        Text with mentions replaced with names
    """

    def replace_mention(match):
        user_id = match.group(1)
        user_name = user_id_to_name_map.get(
            user_id, user_id
        )  # Fallback to user_id if not found
        return f"<@{user_name}>"

    return re.sub(r"<@([A-Z0-9]+)>", replace_mention, text)


def _get_message_id(channel_name: str, ts: str) -> str:
    """
    Generate a unique message ID based on channel name and timestamp.
    """
    return f"{channel_name}_{ts}"
