import argparse
import litellm
from pydantic import BaseModel
from inference.user_role import UserRole, get_user_roles
from utils import file_cache
import config


NORMALIZED_USER_ROLES_PROMPT = """
You are a corporate organization researcher. 

You are given a list of employees information with the following fields enclosed in <user_roles> tags:
- name
- title
- project
- reason
- is_external

The title, project, and is_external fields are automatically inferred by an LLM from
combing through slack messages. Because of the nature of the process, the titles and projects
ended up being too specific and inconsistent. Your job is to unify and normalize these titles
and projects to a more general and consistent format. Some guidelines:
- Use generic titles and terms when possible. For example:
  - Human Resources encompasses Recruiting, people ops, and on-boarding
  - Marketing encompasses Branding, Content, and PR
  - Customer Support encompasses Support, Customer Success, and Customer Service
  - Engineering encompasses Software Engineering, Site Reliability Engineering, and Infrastructure Engineering
- We still want to keep the Managerial titles though, so titles such as "Engineering Manager" should
  be separate from Engineer.
- Try to normalize and combine projects to the broader product or service as well rather than specific projects.


<user_roles>
{user_roles}
</user_roles>
"""


class UserRoleList(BaseModel):
    user_roles: list[UserRole]


@file_cache(f"{config.INFERENCE_DATA_ROOT}/normalized_user_roles.json")
def get_normalized_user_roles() -> list[UserRole]:
    user_roles = [u for u in get_user_roles() if not u.is_external]
    user_roles_str = "\n---\n".join(
        "\n".join([
            f"Name: {ur.name}",
            f"Title: {ur.title}",
            f"Project: {ur.project}",
            f"Reason: {ur.reason}",
        ]) for ur in user_roles
    )
    prompt = NORMALIZED_USER_ROLES_PROMPT.format(user_roles=user_roles_str)
    response = litellm.completion(
        model=config.DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=UserRoleList,
        metadata={"trace_name": "normalize_user_roles"},
    )
    return UserRoleList.model_validate_json(
        response["choices"][0]["message"]["content"]
    ).user_roles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize user roles")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force regeneration of cached inference results",
    )
    args = parser.parse_args()
    get_normalized_user_roles(force_refresh=args.force_refresh)