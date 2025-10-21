from concurrent.futures import ThreadPoolExecutor
import json
import os

import litellm
from pydantic import BaseModel
from tqdm import tqdm

import config
from inference.user_role import UserRole, get_user_roles


USER_MANAGERS_CACHE = f"{config.FEATURES_DATA_ROOT}/user_managers.json"


USER_MANAGER_PROMPT = """
You are a corporate organization researcher.

You are given:
- Information on an employee 
- A list of possible manager

Your goal is to determine the most likely manager for the employee.

A few notes:
- If the employee likely has no manager (e.g. CEO), then output null for the manager.
- If there are multiple possible managers, then output the manager with the highest confidence.

The response should be in a json format, example:
{{"name": "Jane Doe", "manager": "John Smith", "reason": "The employee works on the team where John Smith is the manager."}}

<employee>
{employee}
</employee>

<possible_managers>
{possible_managers}
</possible_managers>
"""


class UserManager(BaseModel):
    name: str
    manager: str | None
    reason: str


def get_user_manager(
    employee: UserRole, possible_managers: list[UserRole]
) -> UserManager:
    employee_str = "\n".join(
        [
            f"Name: {employee.name}",
            f"Title: {employee.title}",
            f"Project: {employee.project}",
        ]
    )
    possible_managers_str = "\n".join(
        [f"- {m.name}: {m.title}, working on {m.project}" for m in possible_managers]
    )
    prompt = USER_MANAGER_PROMPT.format(
        employee=employee_str,
        possible_managers=possible_managers_str,
    )
    response = litellm.completion(
        model="openai/gpt-5",
        messages=[{"role": "user", "content": prompt}],
        response_format=UserManager,
    )
    return UserManager.model_validate_json(response["choices"][0]["message"]["content"])


def get_user_managers(use_cache: bool = True) -> list[UserManager]:
    if use_cache and os.path.exists(USER_MANAGERS_CACHE):
        with open(USER_MANAGERS_CACHE, "r") as f:
            return [UserManager.model_validate(r) for r in json.load(f)]

    user_roles = get_user_roles()
    with ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT_WORKERS) as executor:
        user_managers = list(
            tqdm(
                executor.map(
                    lambda user_role: get_user_manager(user_role, user_roles),
                    user_roles,
                ),
                total=len(user_roles),
                desc="Extracting user managers",
                unit="user",
            )
        )

    with open(USER_MANAGERS_CACHE, "w") as f:
        json.dump([r.model_dump() for r in user_managers], f)
    return user_managers
