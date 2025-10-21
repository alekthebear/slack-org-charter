import argparse
from concurrent.futures import ThreadPoolExecutor

import litellm
from tqdm import tqdm
from pydantic import BaseModel

from features.channel_summaries import get_channel_naming_conventions
from features.user_features import get_user_features
from features.web_search import get_web_search_employees_info
from utils import file_cache
import config


USER_ROLE_PROMPT = """
You are a corporate organization researcher. Your goal is to determine the title/role
for each employee in a given list.

You have the following sources of information:
- The company name, enclosed in <company_name> tags
- The target employee's name, title, and when they joined the company from their slack 
  profile, enclosed in <employee> tags
- Public information on the company's employees on the web, enclosed in 
  <public_employee_info> tags (note: this is possibly outdated and unlikely to be comprehensive)
- Observed slack channel naming conventions, enclosed in <channel_naming_conventions> tags
- Top slack channels that the employee has RECENTLY participated in, with the channel
  name and the number of messages from the employee, enclosed in <recent_channels_list> tags
- Top slack channels that the employee has EVER participated in, with the channel name
  and the number of messages from the employee, enclosed in <all_time_channels_list> tags


Output example:
{{
    "name": "John Doe",
    "title": "Software Engineer",
    "project": "Project X",
    "reason": "The employee participates mainly in engineering channels on the project X"
}}

---
<company_name>
{company_name}
</company_name>

<employee>
{employee}
</employee>

<public_employee_info>
{public_employee_info}
</public_employee_info>

<channel_naming_conventions>
{channel_naming_conventions}
</channel_naming_conventions>

<recent_channels_list>
{recent_channels_list}
</recent_channels_list>

<all_time_channels_list>
{all_time_channels_list}
</all_time_channels_list>
"""


class UserRole(BaseModel):
    name: str
    title: str
    project: str
    reason: str


def get_user_role(user_info_dict: dict, top_n_channels: int = 20) -> UserRole:
    employee_str = "\n".join(
        [
            f"Name: {user_info_dict['full_name']}",
            f"Title: {user_info_dict['title']}",
            f"Joined: {user_info_dict['first_message_time']}",
        ]
    )
    public_info_str = "\n".join(
        [
            f"- {e.name}: {e.role}"
            for e in get_web_search_employees_info(config.COMPANY_NAME).employees
        ]
    )
    recent_channels_list_str = _get_channels_list_str(
        user_info_dict["recent_channels_participation_list"], top_n_channels
    )
    all_time_channels_list_str = _get_channels_list_str(
        user_info_dict["channels_participation_list"], top_n_channels
    )
    prompt = USER_ROLE_PROMPT.format(
        company_name=config.COMPANY_NAME,
        employee=employee_str,
        public_employee_info=public_info_str,
        channel_naming_conventions=get_channel_naming_conventions(),
        recent_channels_list=recent_channels_list_str,
        all_time_channels_list=all_time_channels_list_str,
    )
    response = litellm.completion(
        model=config.DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=UserRole,
    )
    return UserRole.model_validate_json(response["choices"][0]["message"]["content"])


@file_cache(f"{config.INFERENCE_DATA_ROOT}/user_roles.json")
def get_user_roles(
    top_n_channels: int = 20,
    max_workers: int = config.MAX_CONCURRENT_WORKERS,
) -> list[UserRole]:
    user_features_df = get_user_features()
    user_dicts = [row.to_dict() for _, row in user_features_df.iterrows()]

    print(f"Processing {len(user_dicts)} users with {max_workers} parallel workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        user_roles = list(
            tqdm(
                executor.map(
                    lambda user_dict: get_user_role(user_dict, top_n_channels),
                    user_dicts,
                ),
                total=len(user_dicts),
                desc="Inferring user roles",
                unit="user",
            )
        )

    return user_roles


def _get_channels_list_str(channel_msg_list: list[dict], top_n_channels: int) -> str:
    return "\n".join(
        [f"- {c['channel']}: {c['cnt']}" for c in channel_msg_list[:top_n_channels]]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate user roles inference")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force regeneration of cached inference results",
    )
    args = parser.parse_args()

    get_user_roles(force_refresh=args.force_refresh)
