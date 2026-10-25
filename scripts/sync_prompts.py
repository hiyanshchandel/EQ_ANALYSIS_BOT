"""Upload the prompts in `agent/prompts.py` to Langfuse as a new version.

    python scripts/sync_prompts.py                 # label as the configured PROMPT_LABEL
    python scripts/sync_prompts.py --label staging

Langfuse keeps every version. The app fetches whichever version carries its label, so
you can try a prompt under "staging" and promote it to "production" in the UI, with no
code change or redeploy.
"""

from __future__ import annotations

import argparse
import sys

from eq_analyst.agent.prompts import LOCAL_PROMPTS
from eq_analyst.core.config import get_settings
from eq_analyst.observability.tracing import init_langfuse


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=settings.prompt_label)
    args = parser.parse_args()

    langfuse = init_langfuse(settings)
    if langfuse is None:
        print("Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY first.")
        return 2

    for name, template in LOCAL_PROMPTS.items():
        prompt = langfuse.create_prompt(
            name=name,
            prompt=template,
            type="text",
            labels=[args.label],
            commit_message="Synced from agent/prompts.py",
        )
        print(f"{name}: version {prompt.version} labelled '{args.label}'")
    langfuse.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
