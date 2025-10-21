import os
import tempfile

import pandas as pd
from pydantic import BaseModel

from utils import file_cache


class SampleModel(BaseModel):
    name: str
    value: int


class TestFileCacheDecorator:
    def test_text_cache(self):
        """Test caching text data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "test_cache.txt")
            call_count = []

            @file_cache(cache_path)
            def get_text() -> str:
                call_count.append(1)
                return "Hello, World!"

            # First call - should execute function
            result1 = get_text()
            assert result1 == "Hello, World!"
            assert len(call_count) == 1
            assert os.path.exists(cache_path)

            # Second call - should use cache
            result2 = get_text()
            assert result2 == "Hello, World!"
            assert len(call_count) == 1  # Function not called again

            # Force refresh - should execute function again
            result3 = get_text(force_refresh=True)
            assert result3 == "Hello, World!"
            assert len(call_count) == 2

    def test_json_cache(self):
        """Test caching JSON data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "test_cache.json")
            call_count = []

            @file_cache(cache_path)
            def get_dict() -> dict:
                call_count.append(1)
                return {"name": "Alice", "age": 30}

            # First call
            result1 = get_dict()
            assert result1 == {"name": "Alice", "age": 30}
            assert len(call_count) == 1

            # Second call - should use cache
            result2 = get_dict()
            assert result2 == {"name": "Alice", "age": 30}
            assert len(call_count) == 1

    def test_pydantic_single_model_cache(self):
        """Test caching a single Pydantic model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "test_cache.json")
            call_count = []

            @file_cache(cache_path)
            def get_model() -> SampleModel:
                call_count.append(1)
                return SampleModel(name="Test", value=42)

            # First call
            result1 = get_model()
            assert isinstance(result1, SampleModel)
            assert result1.name == "Test"
            assert result1.value == 42
            assert len(call_count) == 1

            # Second call - should use cache
            result2 = get_model()
            assert isinstance(result2, SampleModel)
            assert result2.name == "Test"
            assert result2.value == 42
            assert len(call_count) == 1

    def test_pydantic_list_cache(self):
        """Test caching a list of Pydantic models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "test_cache.json")
            call_count = []

            @file_cache(cache_path)
            def get_models() -> list[SampleModel]:
                call_count.append(1)
                return [
                    SampleModel(name="Model1", value=1),
                    SampleModel(name="Model2", value=2),
                ]

            # First call
            result1 = get_models()
            assert isinstance(result1, list)
            assert len(result1) == 2
            assert all(isinstance(m, SampleModel) for m in result1)
            assert result1[0].name == "Model1"
            assert result1[1].value == 2
            assert len(call_count) == 1

            # Second call - should use cache
            result2 = get_models()
            assert isinstance(result2, list)
            assert len(result2) == 2
            assert len(call_count) == 1

    def test_parquet_cache(self):
        """Test caching pandas DataFrame as parquet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "test_cache.parquet")
            call_count = []

            @file_cache(cache_path)
            def get_dataframe() -> pd.DataFrame:
                call_count.append(1)
                return pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

            # First call
            result1 = get_dataframe()
            assert isinstance(result1, pd.DataFrame)
            assert len(result1) == 3
            assert list(result1.columns) == ["col1", "col2"]
            assert len(call_count) == 1

            # Second call - should use cache
            result2 = get_dataframe()
            assert isinstance(result2, pd.DataFrame)
            pd.testing.assert_frame_equal(result1, result2)
            assert len(call_count) == 1

    def test_force_refresh_overwrites_cache(self):
        """Test that force_refresh=True overwrites the cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "test_cache.txt")
            counter = [0]

            @file_cache(cache_path)
            def get_text() -> str:
                counter[0] += 1
                return f"Call {counter[0]}"

            # First call
            result1 = get_text()
            assert result1 == "Call 1"

            # Second call - uses cache
            result2 = get_text()
            assert result2 == "Call 1"  # From cache

            # Force refresh - executes function again
            result3 = get_text(force_refresh=True)
            assert result3 == "Call 2"

            # Fourth call - uses new cache
            result4 = get_text()
            assert result4 == "Call 2"  # From updated cache

    def test_function_with_parameters(self):
        """Test that the decorator works with functions that have parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "test_cache.txt")

            @file_cache(cache_path)
            def greet(name: str, greeting: str = "Hello") -> str:
                return f"{greeting}, {name}!"

            result = greet("Alice")
            assert result == "Hello, Alice!"

            result2 = greet("Bob", greeting="Hi")
            # Note: Since cache_path is the same, it will return cached value
            assert result2 == "Hello, Alice!"

    def test_cache_directory_creation(self):
        """Test that cache directories are created automatically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "subdir", "nested", "test_cache.txt")

            @file_cache(cache_path)
            def get_text() -> str:
                return "Test"

            result = get_text()
            assert result == "Test"
            assert os.path.exists(cache_path)
            assert os.path.exists(os.path.dirname(cache_path))

    def test_corrupted_cache_fallback(self):
        """Test that corrupted cache falls back to executing the function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "test_cache.json")

            @file_cache(cache_path)
            def get_data() -> dict:
                return {"key": "value"}

            # First call - creates cache
            result1 = get_data()
            assert result1 == {"key": "value"}

            # Corrupt the cache
            with open(cache_path, "w") as f:
                f.write("invalid json {{{")

            # Should handle corruption gracefully and re-execute
            result2 = get_data()
            assert result2 == {"key": "value"}

