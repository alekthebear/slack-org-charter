import litellm
import os

if os.getenv("LANGFUSE_SECRET_KEY") is not None:
    litellm.success_callback = ["langfuse_otel"]
    litellm.failure_callback = ["langfuse_otel"]
    print("LiteLLM to Langfuse instrumentation enabled")
