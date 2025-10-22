import argparse
from concurrent.futures import ThreadPoolExecutor

import litellm
from pydantic import BaseModel
from tqdm import tqdm

import config
from features.mention_graph import get_user_mention_graph
from features.user_features import get_user_features
from inference.user_role import UserRole, get_user_roles
from utils import file_cache


USER_MANAGER_PROMPT = """
You are a corporate organization researcher.

You are given:
- Information on an employee 
- A list of other employees in the organization
- Mention relationships showing how often the employee mentions other employees and vice versa
  (derived from recent Slack messages)

Your goal is to determine the most likely manager for the employee.

A few notes:
- If the employee likely has no manager (e.g. CEO), then output null for the manager.
- If there are multiple possible managers, then output the manager with the highest confidence.
- The employee's title is the strongest indicator of organizational hierarchy. Generally engineers report to
  engineering managers, product managers report to senior product managers, etc. It’s uncommon for employees 
  from different disciplines to report directly to one another.
- The employee's project and activities are also helpful in determining the employee's relationships
- Mention patterns can be useful: employees often mention their managers more frequently
  than their managers mention them, though this isn't always the case.

The response should be in a json format, example:
{{"name": "Jane Doe", "manager": "John Smith", "reason": "The employee works on the team where John Smith is the manager."}}

<employee>
{employee}
</employee>

<possible_managers>
{possible_managers}
</possible_managers>

<mention_relationships>
{mention_relationships}
</mention_relationships>
"""


class UserManager(BaseModel):
    name: str
    manager: str | None
    reason: str


def get_mention_stats(
    employee_name: str,
    mention_graph: list[dict],
) -> str:
    """
    Compute bidirectional mention statistics between an employee and potential managers.

    Args:
        employee_name: Full name of the employee
        mention_graph: List of mention relationships

    Returns:
        Formatted string with mention statistics for each manager
    """
    lines = []
    employee_mentions = {}
    mentions_employee = {}
    for mention in mention_graph:
        if mention["user_name"] == mention["mentions"]:
            continue
        elif mention["user_name"] == employee_name:
            employee_mentions[mention["mentions"]] = mention["count"]
        elif mention["mentions"] == employee_name:
            mentions_employee[mention["user_name"]] = mention["count"]
    
    coworker_names = set(employee_mentions.keys()) | set(mentions_employee.keys())
    for coworker in coworker_names:
        lines.append(
            f"- {employee_name} mentions {coworker} {employee_mentions.get(coworker, 0)} times, "
            f"{coworker} mentions {employee_name} {mentions_employee.get(coworker, 0)} times"
        )
    return "\n".join(lines)


def _get_employee_str(employee: UserRole) -> str:
    return "\n".join(
        [
            f"Name: {employee.name}",
            f"Title: {employee.title}",
            f"Project: {employee.project}",
            f"Activities: {employee.reason}",
        ]
    )

def get_user_manager(
    employee: UserRole,
    possible_managers: list[UserRole],
    mention_graph: list[dict],
) -> UserManager:
    employee_str = _get_employee_str(employee)
    possible_managers_str = "\n---\n".join(
        [_get_employee_str(m) for m in possible_managers if m.name != employee.name]
    )
    mention_relationships_str = get_mention_stats(employee.name, mention_graph)
    prompt = USER_MANAGER_PROMPT.format(
        employee=employee_str,
        possible_managers=possible_managers_str,
        mention_relationships=mention_relationships_str,
    )
    response = litellm.completion(
        model=config.DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=UserManager,
        metadata={"trace_name": "user_manager"},
    )
    return UserManager.model_validate_json(response["choices"][0]["message"]["content"])


@file_cache(f"{config.INFERENCE_DATA_ROOT}/user_managers.json")
def get_user_managers() -> list[UserManager]:
    user_roles = [u for u in get_user_roles() if not u.is_external]

    # Load mention graph and name-to-ID mapping once
    mention_graph = get_user_mention_graph()
    user_features_df = get_user_features()

    with ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT_WORKERS) as executor:
        user_managers = list(
            tqdm(
                executor.map(
                    lambda user_role: get_user_manager(
                        user_role, user_roles, mention_graph
                    ),
                    user_roles,
                ),
                total=len(user_roles),
                desc="Inferring user managers",
                unit="user",
            )
        )
    return user_managers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate user managers inference")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force regeneration of cached inference results",
    )
    args = parser.parse_args()

    get_user_managers(force_refresh=args.force_refresh)
