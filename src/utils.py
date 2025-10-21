from functools import cache, wraps
import inspect
import json
import os
from typing import get_args, get_origin

import pandas as pd
from pydantic import BaseModel

import config


# ====================
#  Development Utils
# ====================
def pretty_print_messages(
    messages_df: pd.DataFrame, return_string: bool = False
) -> str | None:
    """For debugging/development purposes, print the messages in a readable format."""
    messages_df = messages_df.sort_values(by=["channel", "timestamp"], ascending=True)
    result = ""
    for channel, channel_messages_df in messages_df.groupby("channel"):
        result += "--------------------------------\n"
        result += f"Channel: {channel}\n"
        result += f"Message Count: {len(channel_messages_df)}\n"
        result += f"Message Range: {channel_messages_df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')} to {channel_messages_df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')}\n"
        result += "--------------------------------\n"
        for _, row in channel_messages_df.iterrows():
            timestamp = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            user_name = row["user_name"]
            text_formatted = row["text_formatted"]
            result += f"({timestamp}) {user_name}: {text_formatted}\n"
        result += "\n"
    if return_string:
        return result
    else:
        print(result)


@cache
def get_user_id_to_name_map(user_file_path: str = config.USERS_FILE_PATH) -> dict:
    with open(user_file_path) as f:
        users = json.load(f)
    return {u["id"]: u["profile"]["real_name_normalized"] for u in users}


# =======================
#  File Caching Decorator
# =======================
def file_cache(cache_path: str):
    """
    Decorator to cache function results to a file.

    Automatically infers the data type from the function's return type annotation:
    - str -> text file
    - dict or list[dict] -> JSON file
    - pd.DataFrame -> Parquet file
    - BaseModel or list[BaseModel] -> JSON file (Pydantic)

    Args:
        cache_path: Path to the cache file

    The decorator automatically supports a `force_refresh` parameter:
    - force_refresh=True: Execute function and overwrite cache
    - force_refresh=False (default): Use cache if available

    Example:
        @file_cache("cache.json")
        def get_data(company_name: str) -> MyModel:
            return MyModel(...)

        # Usage:
        get_data("Acme")  # Uses cache if available
        get_data("Acme", force_refresh=True)  # Forces refresh
    """

    def decorator(func):
        # Get the return type annotation
        sig = inspect.signature(func)
        return_annotation = sig.return_annotation

        # Infer data type from return annotation
        data_type = _infer_data_type(return_annotation)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract and remove force_refresh from kwargs
            force_refresh = kwargs.pop("force_refresh", False)

            if not force_refresh and os.path.exists(cache_path):
                try:
                    print(f"Loading from cache: {cache_path}")
                    if data_type == "text":
                        with open(cache_path, "r") as f:
                            return f.read()
                    elif data_type == "json":
                        with open(cache_path, "r") as f:
                            return json.load(f)
                    elif data_type == "pydantic":
                        with open(cache_path, "r") as f:
                            data = json.load(f)

                        # Determine if return type is a list or single model
                        origin = get_origin(return_annotation)
                        if origin is list:
                            # Get the model class from list[Model]
                            model_class = get_args(return_annotation)[0]
                            return [model_class.model_validate(item) for item in data]
                        else:
                            # Single model
                            return return_annotation.model_validate(data)

                    elif data_type == "parquet":
                        return pd.read_parquet(cache_path)

                except Exception as e:
                    # If cache loading fails, proceed to execute function
                    print(f"Warning: Failed to load cache from {cache_path}: {e}")

            # Execute the function
            result = func(*args, **kwargs)

            # Save to cache
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)

                print(f"Saving to cache: {cache_path}")
                if data_type == "text":
                    with open(cache_path, "w") as f:
                        f.write(result)
                elif data_type == "json":
                    with open(cache_path, "w") as f:
                        json.dump(result, f, indent=2)
                elif data_type == "pydantic":
                    with open(cache_path, "w") as f:
                        # Handle both single model and list of models
                        if isinstance(result, list):
                            json.dump(
                                [item.model_dump() for item in result], f, indent=2
                            )
                        else:
                            json.dump(result.model_dump(), f, indent=2)
                elif data_type == "parquet":
                    result.to_parquet(cache_path)

            except Exception as e:
                print(f"Warning: Failed to save cache to {cache_path}: {e}")

            return result

        return wrapper

    return decorator


def _infer_data_type(return_annotation) -> str:
    """
    Infer the cache data type from a function's return type annotation.

    Returns one of: "text", "json", "pydantic", "parquet"
    """
    # Handle None/missing annotation
    if return_annotation is inspect.Parameter.empty:
        raise ValueError(
            "Function must have a return type annotation for file_cache decorator"
        )

    # Get the origin type (e.g., list from list[Model])
    origin = get_origin(return_annotation)

    if return_annotation is str:
        return "text"
    if return_annotation is pd.DataFrame:
        return "parquet"
    if return_annotation is dict:
        return "json"
    if origin is list:
        args = get_args(return_annotation)
        if args:
            inner_type = args[0]
            # list[dict] -> json
            if inner_type is dict:
                return "json"
            # list[BaseModel] -> pydantic
            if isinstance(inner_type, type) and issubclass(inner_type, BaseModel):
                return "pydantic"
    if isinstance(return_annotation, type) and issubclass(return_annotation, BaseModel):
        return "pydantic"

    # Default fallback
    raise ValueError(
        f"Cannot infer cache type from return annotation: {return_annotation}. "
        f"Supported types: str, dict, list[dict], pd.DataFrame, BaseModel, list[BaseModel]"
    )
