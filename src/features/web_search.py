import argparse

import litellm
from pydantic import BaseModel

import config
from utils import file_cache


WEB_SEARCH_PROMPT = """
You are a corporate organization researcher.

Search the web to find who are the current employees at the company {company_name} and what are their roles?

Only return the list of json objects (jsonl format), do not include any other text in your response.
"""


class WebSearchEmployeeInfo(BaseModel):
    name: str
    role: str
    source_urls: list[str]


class WebSearchResults(BaseModel):
    employees: list[WebSearchEmployeeInfo]


@file_cache(f"{config.FEATURES_DATA_ROOT}/web_search.json")
def get_web_search_employees_info(company_name: str = config.COMPANY_NAME) -> WebSearchResults:
    prompt = WEB_SEARCH_PROMPT.format(company_name=company_name)
    response = litellm.completion(
        model=config.WEB_SEARCH_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=WebSearchResults,
        metadata={"trace_name": "web_search_employees_info"},
    )
    return WebSearchResults.model_validate_json(
        response["choices"][0]["message"]["content"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate web search employee info features"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force regeneration of cached features",
    )
    args = parser.parse_args()

    get_web_search_employees_info(
        config.COMPANY_NAME, force_refresh=args.force_refresh
    )
