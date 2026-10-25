"""Telegram handlers and application lifecycle (python-telegram-bot v22, fully async)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager, suppress

from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    LinkPreviewOptions,
    Update,
)
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from eq_analyst.agent.checkpoint import sqlite_checkpointer
from eq_analyst.agent.nodes.replies import HELP_TEXT
from eq_analyst.bootstrap import build_agent_runtime
from eq_analyst.bot.formatting import format_message, format_report
from eq_analyst.core.config import Settings
from eq_analyst.core.log import get_logger
from eq_analyst.observability.tracing import init_langfuse
from eq_analyst.service import AnalystService

log = get_logger(__name__)

ERROR_TEXT = "Sorry, something went wrong on my side. Please try again in a minute."
MAX_MESSAGE_CHARS = 2000  # analyses don't need essays; also caps LLM cost per message


# --- lifecycle --------------------------------------------------------------------


async def on_startup(app: Application) -> None:
    """Runs once inside the bot's event loop, before polling starts."""
    settings: Settings = app.bot_data["settings"]
    stack = AsyncExitStack()
    checkpointer = await stack.enter_async_context(sqlite_checkpointer(settings.checkpoint_db_path))
    langfuse = init_langfuse(settings)
    runtime = await build_agent_runtime(settings, checkpointer, langfuse)

    app.bot_data["exit_stack"] = stack
    app.bot_data["langfuse"] = langfuse
    app.bot_data["service"] = AnalystService(runtime, settings, langfuse)

    await app.bot.set_my_commands(
        [
            BotCommand("help", "How to use this bot"),
            BotCommand("reset", "Forget this conversation"),
        ]
    )
    log.info("bot_started", model=settings.gemini_model, tracing=langfuse is not None)


async def on_shutdown(app: Application) -> None:
    if langfuse := app.bot_data.get("langfuse"):
        langfuse.shutdown()  # flush buffered traces before exiting
    if stack := app.bot_data.get("exit_stack"):
        await stack.aclose()  # closes the SQLite connection
    log.info("bot_stopped")


# --- helpers ----------------------------------------------------------------------


def _service(context: ContextTypes.DEFAULT_TYPE) -> AnalystService:
    return context.bot_data["service"]


def _is_allowed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    allowed: list[int] = context.bot_data["settings"].allowed_chat_ids
    chat = update.effective_chat
    return not allowed or (chat is not None and chat.id in allowed)


@asynccontextmanager
async def keep_typing(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> AsyncIterator[None]:
    """Show "typing..." for the whole analysis (Telegram clears it after ~5 seconds)."""

    async def loop() -> None:
        while True:
            await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
            await asyncio.sleep(4)

    task = asyncio.create_task(loop())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


def feedback_keyboard(trace_id: str) -> InlineKeyboardMarkup:
    # callback_data is limited to 64 bytes: "fb:1:" + 32-char trace id fits.
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("👍 Helpful", callback_data=f"fb:1:{trace_id}"),
                InlineKeyboardButton("👎 Not helpful", callback_data=f"fb:0:{trace_id}"),
            ]
        ]
    )


# --- handlers ---------------------------------------------------------------------


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message and _is_allowed(update, context):
        await update.effective_message.reply_text("Hi! " + HELP_TEXT)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message and _is_allowed(update, context):
        await update.effective_message.reply_text(HELP_TEXT)


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not (update.effective_message and update.effective_chat and _is_allowed(update, context)):
        return
    await _service(context).reset_conversation(update.effective_chat.id)
    await update.effective_message.reply_text("Done. I've forgotten our conversation.")


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message, chat, user = update.effective_message, update.effective_chat, update.effective_user
    if not (message and message.text and chat and user) or not _is_allowed(update, context):
        return
    if len(message.text) > MAX_MESSAGE_CHARS:
        await message.reply_text("That message is too long. Please keep it under 2000 characters.")
        return

    try:
        async with keep_typing(context, chat.id):
            result = await _service(context).handle_message(
                chat_id=chat.id, user_id=user.id, text=message.text
            )
    except Exception:
        log.exception("turn_failed", chat_id=chat.id)
        await message.reply_text(ERROR_TEXT)
        return

    body = format_report(result.report) if result.report else format_message(result.text)
    await message.reply_text(
        body,
        parse_mode=ParseMode.HTML,
        reply_markup=feedback_keyboard(result.trace_id) if result.trace_id else None,
        link_preview_options=LinkPreviewOptions(is_disabled=True),
    )


async def on_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None or query.data is None or update.effective_user is None:
        return
    _, value, trace_id = query.data.split(":", 2)
    _service(context).record_feedback(
        trace_id=trace_id, positive=value == "1", user_id=update.effective_user.id
    )
    await query.answer("Thanks for the feedback!")
    await query.edit_message_reply_markup(reply_markup=None)  # one vote per message


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.error("telegram_error", error=str(context.error), exc_info=context.error)


# --- factory ----------------------------------------------------------------------


def build_application(settings: Settings) -> Application:
    if settings.telegram_bot_token is None:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set. Add it to your .env file.")

    app = (
        Application.builder()
        .token(settings.telegram_bot_token.get_secret_value())
        .concurrent_updates(True)  # one slow analysis must not block other users
        .post_init(on_startup)
        .post_shutdown(on_shutdown)
        .build()
    )
    app.bot_data["settings"] = settings

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(CallbackQueryHandler(on_feedback, pattern=r"^fb:[01]:"))
    app.add_error_handler(on_error)
    return app
