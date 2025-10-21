import litellm
import pandas as pd
import os

from config import FEATURES_DATA_ROOT
from extract.channels import get_channels


CHANNEL_CONVENTION_CACHE = f"{FEATURES_DATA_ROOT}/channel_conventions.txt"

CHANNEL_INPUT_FORMAT = """
You are given an input list of slack channels from the same workspace with the following format, separated by "---":
- Channel: <channel_name>
- (Optional) Description: <channel_description>
"""


CHANNEL_NAMING_CONVENTIONS_PROMPT = """
You are a corporate organization researcher. 
{channel_input_prompt}

Describe some of the naming conventions/patterns that you can find in the channel names that may be useful
for someone who is trying to understand how slack is used in the company. 

Keep your response focused and crisp:
- Focus only listing and describing the patterns
- Useful to put high signal patterns in a section different from low signal ones
- No need to offer generic insights such as "the company is very collaborative" or "the company is very formal"

Here are the list of channels:
<channels>
{channel_list_str}
</channels>

"""


def get_channel_naming_conventions(use_cache: bool = True) -> str:
    if use_cache and os.path.exists(CHANNEL_CONVENTION_CACHE):
        with open(CHANNEL_CONVENTION_CACHE, "r") as f:
            return f.read()

    channels_df = get_channels()
    channel_list_str = _stringify_channels_input(channels_df)
    prompt = CHANNEL_NAMING_CONVENTIONS_PROMPT.format(
        channel_input_prompt=CHANNEL_INPUT_FORMAT, channel_list_str=channel_list_str
    )
    response = litellm.completion(
        model="openai/gpt-5",
        messages=[{"content": prompt, "role": "user"}],
    )
    result = response.choices[0].message.content

    with open(CHANNEL_CONVENTION_CACHE, "w") as f:
        f.write(result)
    return result


def _format_channel_rows(channel_row):
    properties = [f"Channel: {channel_row['name']}"]

    # fill in description if purpose or topic is not empty
    if channel_row["purpose"] != "":
        properties.append(f"Description: {channel_row['purpose'].replace('\n', ' ')}")
    elif channel_row["topic"] != "":
        properties.append(f"Description: {channel_row['topic'].replace('\n', ' ')}")
    return "\n".join(properties)


def _stringify_channels_input(channels_df: pd.DataFrame):
    return "\n---\n".join(
        sorted(list(channels_df.apply(_format_channel_rows, axis=1).tolist()))
    )
