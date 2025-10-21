import argparse
import json

import litellm
from tqdm.auto import tqdm
import pandas as pd

from config import DEFAULT_MODEL, FEATURES_DATA_ROOT
from extract.channels import get_channels
from features.channel_conventions import (
    CHANNEL_INPUT_FORMAT,
    get_channel_naming_conventions,
    _stringify_channels_input,
)
from utils import file_cache


CHANNELS_USEFULNESS_LABEL_PROMPT = """
You are a corporate organization researcher. Your goal is to understand:
1. the organizational hierarchy
2. employee projects/title within the company

Your plan to investigate and read through messages in key channels to achieve your goal.
In order to determine which channels are important to your goal, you will scan through the channel names and descriptions
to determine which channels are likely to be useful.

{channel_input_prompt}

You have also noted the following channel naming conventions:
<naming_conventions>
{channel_naming_conventions}
</naming_conventions>

For each the channels provided, label them as one of the following categories: useful, maybe, useless. Some examples:
- "social" or non-work related channels are unlikely to be useful for understanding who is a manager and what people are working on.
- "team-<team_name>" or "project-<project_name>" channels can be very useful for understanding the team/project structure
- Channels filled with auto-generated alerts are unlikely to be useful
- Channels that deal with approval processes can be useful for classifying managers

Output (and only output) list of JSON objects, each containing the following fields:
- name: the name of the channel
- usefulness: the usefulness label of the channel
- reason: the reason for the usefulness label

Example Output:
[
 {{"name": "social", "usefulness": "useless", "reason": "This channel is used for socializing, no professional signals."}},
 {{"name": "engineering", "usefulness": "maybe", "reason": "This channel could be useful for understanding the engineering team structure."}},
 {{"name": "team-feature-<feature_name>", "usefulness": "useful", "reason": "This channel is likely to be useful for understanding the feature team structure."}}
]

Here are the list of channels to label:
---
{channel_list_str}
---
"""


@file_cache(f"{FEATURES_DATA_ROOT}/channel_usefulness.json")
def get_channel_usefulness_labels(
    channels_df: pd.DataFrame, batch_size: int = 100
) -> list[dict]:
    """Get the usefulness labels for a list of channels."""
    channel_naming_conventions = get_channel_naming_conventions()
    all_results = []

    with tqdm(total=len(channels_df), desc="Labeling channels") as pbar:
        for i in tqdm(total=len(channels_df), desc="Labeling channels"):
            batch_df = channels_df.iloc[i : i + batch_size]
            batch_results = _label_channels_batch(batch_df, channel_naming_conventions)
            all_results.extend(batch_results)
            pbar.update(len(batch_df))

    return all_results


def _label_channels_batch(
    channel_batch_df: pd.DataFrame, channel_naming_conventions: str
) -> list[dict]:
    """Label a batch of channels for usefulness using LLM."""
    channel_list_str = _stringify_channels_input(channel_batch_df)
    prompt = CHANNELS_USEFULNESS_LABEL_PROMPT.format(
        channel_input_prompt=CHANNEL_INPUT_FORMAT,
        channel_list_str=channel_list_str,
        channel_naming_conventions=channel_naming_conventions,
    )
    response = litellm.completion(
        model=DEFAULT_MODEL,
        messages=[{"content": prompt, "role": "user"}],
    )
    result = response.choices[0].message.content
    try:
        parsed_result = json.loads(result)
        assert len(parsed_result) == len(channel_batch_df)
        return parsed_result
    except json.JSONDecodeError:
        raise ValueError(f"Error parsing JSON: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate channel usefulness labels features"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force regeneration of cached features",
    )
    args = parser.parse_args()

    channels_df = get_channels()
    get_channel_usefulness_labels(channels_df, force_refresh=args.force_refresh)
