"""Entry point: `python -m eq_analyst.bot` (or the `eq-bot` script)."""

from __future__ import annotations

from telegram import Update

from eq_analyst.bot.app import build_application
from eq_analyst.core.config import get_settings
from eq_analyst.core.log import configure_logging


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    # Long polling: the bot asks Telegram for new messages. Simple, and works behind NAT.
    # At larger scale you'd switch to webhooks (Telegram pushes to an HTTPS endpoint).
    build_application(settings).run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
