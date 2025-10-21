import json
import os
import litellm
from pydantic import BaseModel

import config


WEB_SEARCH_CACHE = f"{config.FEATURES_DATA_ROOT}/web_search.json"


WEB_SEARCH_PROMPT = """
You are a corporate organization researcher.

Who are the current employees at the company {company_name} and what are their roles?

The response should be in a list json format, example:
[
    {{"name": "John Doe", "role": "CEO", "source_urls": ["https://www.company_url.com/company/executives"]}},
    {{"name": "Jane Smith", "role": "CTO", "source_urls": ["https://www.company_url.com/company/executives"]}}
]

Only return the list of json objects (jsonl format), do not include any other text in your response.
"""


class WebSearchEmployeeInfo(BaseModel):
    name: str
    role: str
    source_urls: list[str]


class WebSearchResults(BaseModel):
    employees: list[WebSearchEmployeeInfo]


def get_web_search_employees_info(
    company_name: str, use_cache: bool = True
) -> WebSearchResults:
    if use_cache and os.path.exists(WEB_SEARCH_CACHE):
        with open(WEB_SEARCH_CACHE, "r") as f:
            return WebSearchResults.model_validate(json.load(f))

    prompt = WEB_SEARCH_PROMPT.format(company_name=company_name)
    response = litellm.completion(
        model="openai/gpt-5-search-api",
        messages=[{"role": "user", "content": prompt}],
        response_format=WebSearchResults,
    )
    employee_roles = WebSearchResults.model_validate_json(
        response["choices"][0]["message"]["content"]
    )
    with open(WEB_SEARCH_CACHE, "w") as f:
        json.dump(employee_roles.model_dump(), f)
    return employee_roles
