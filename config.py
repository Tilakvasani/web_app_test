"""
Universal Configuration Module for the Model Context Protocol (MCP) Backend.

Provides standard file registries, directory initializations, and centralized helpers
for initializing Azure OpenAI models and generating multi-server connection configurations.
"""

import os
import logging
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

# Load env variables on startup
load_dotenv()

# File Registries & Fallbacks
STATE_FILE = "mcp_state.json"
OAUTH_FLOWS_FILE = "oauth_flows.json"
UPLOAD_DIR = "uploaded_files"

# Create secure directories if not present
os.makedirs(UPLOAD_DIR, exist_ok=True)


_cfg_logger = logging.getLogger("mcp_backend")


# LLM Deployment Setup
def get_llm() -> AzureChatOpenAI:
    """
    Initializes and returns the primary Azure OpenAI Chat model client.

    Retrieves connection endpoints, API keys, and deployment names from
    the active environment variables, with support for streaming responses.

    Returns:
        AzureChatOpenAI: Instantiated LangChain-compliant Azure OpenAI chat model client.
    """
    deployment = (
        os.getenv("AZURE_OPENAI_DEPLOYMENT") or
        os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    )
    if not deployment:
        # Bug #18 fix: the hard-coded default "gpt-4.1-mini" will not exist on
        # most Azure instances. Log a clear warning so it's obvious at startup.
        _cfg_logger.warning(
            "[CONFIG] ⚠️  Neither AZURE_OPENAI_DEPLOYMENT nor AZURE_OPENAI_DEPLOYMENT_NAME "
            "is set in your .env file. Falling back to 'gpt-4.1-mini' — this deployment "
            "almost certainly does not exist on your Azure instance and every LLM call "
            "will fail with a 404. Please set AZURE_OPENAI_DEPLOYMENT in your .env."
        )
        deployment = "gpt-4.1-mini"

    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        azure_deployment=deployment,
        streaming=True,
    )


def get_cheap_llm() -> AzureChatOpenAI:
    """
    Returns a cheaper, non-streaming LLM instance for lightweight classification
    tasks (sifter, planner, history tone detection) that don't need the full
    primary model.

    Falls back to the primary deployment if AZURE_OPENAI_CHEAP_DEPLOYMENT is
    not set — still saves cost by disabling streaming on these short calls.
    """
    cheap_deployment = (
        os.getenv("AZURE_OPENAI_CHEAP_DEPLOYMENT") or
        os.getenv("AZURE_OPENAI_DEPLOYMENT") or
        os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or
        "gpt-4.1-mini"
    )
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        azure_deployment=cheap_deployment,
        streaming=False,   # no streaming needed for classification/planning
        max_tokens=512,    # sifter/planner outputs are always short
    )

from typing import Dict, Any

def _make_cfg(url: str, auth_type: str, auth_value: str, transport: str = "streamable_http") -> Dict[str, Any]:
    """
    Constructs a standardized configuration dictionary for an MCP server connection.

    Generates the appropriate structure with auth headers (Bearer token, API Key, or none)
    and maps the specified communication transport protocol.

    Args:
        url (str): The HTTP/SSE endpoint URL of the target MCP server.
        auth_type (str): The authentication method ('bearer', 'api_key', or 'none').
        auth_value (str): The raw authorization token or API key credentials.
        transport (str, optional): Communication channel type. Defaults to "streamable_http".

    Returns:
        Dict[str, Any]: A complete client connection configuration block for MultiServerMCPClient.
    """
    cfg: Dict[str, Any] = {"url": url, "transport": transport}
    if auth_type == "bearer":
        cfg["headers"] = {"Authorization": f"Bearer {auth_value}"}
    elif auth_type == "api_key":
        cfg["headers"] = {"X-API-Key": auth_value}
    return cfg