from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    firecrawl_api_key: str = os.getenv("FIRECRAWL_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_chat_model: str = os.getenv("GROQ_CHAT_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

    # Firecrawl MCP server runtime configuration.
    firecrawl_mcp_command: str = os.getenv("FIRECRAWL_MCP_COMMAND", "npx")
    firecrawl_mcp_args: str = os.getenv("FIRECRAWL_MCP_ARGS", "-y,firecrawl-mcp")
    firecrawl_mcp_timeout_seconds: int = int(os.getenv("FIRECRAWL_MCP_TIMEOUT_SECONDS", "120"))
    firecrawl_max_pages: int = int(os.getenv("FIRECRAWL_MAX_PAGES", "100"))

    handbook_root_url: str = os.getenv("HANDBOOK_ROOT_URL", "https://handbook.gitlab.com/handbook/")
    direction_url: str = os.getenv(
        "DIRECTION_URL",
        "https://about.gitlab.com/releases/whats-new/#whats-coming",
    )

    project_root: Path = Path(__file__).resolve().parent.parent

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def seed_dir(self) -> Path:
        return self.data_dir / "seed_urls"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def firecrawl_mcp_args_list(self) -> list[str]:
        return [part.strip() for part in self.firecrawl_mcp_args.split(",") if part.strip()]


settings = Settings()
