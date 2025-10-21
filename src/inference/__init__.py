import litellm
import os

if os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_HOST"):
    litellm.success_callback = ["langfuse_otel"]
    litellm.failure_callback = ["langfuse_otel"]
    print("LiteLLM to Langfuse instrumentation enabled")
