"""Entry point: `python -m eq_analyst.mcp_server` (or the `eq-mcp-server` script)."""

from __future__ import annotations

from eq_analyst.core.config import get_settings
from eq_analyst.core.log import configure_logging, get_logger
from eq_analyst.mcp_server.server import create_server


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    get_logger(__name__).info(
        "mcp_server_starting", host=settings.mcp_host, port=settings.mcp_port, path="/mcp"
    )
    create_server(settings).run(transport="streamable-http")


if __name__ == "__main__":
    main()
