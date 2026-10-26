"""Ask the assistant from the terminal:  python -m app.cli "Can I return my X2 after 20 days?" """

import sys

from app.config import load_settings
from app.factory import build_assistant


def main() -> None:
    message = " ".join(sys.argv[1:]) or input("You: ")
    reply = build_assistant(load_settings()).answer(message)
    print(f"\nBolt: {reply.text}\n")
    for step in reply.trace:
        print(f"  [{step.status:7}] {step.name:14} {step.ms:7.0f} ms  {step.detail}")
    print(f"\n  total {reply.total_ms:.0f} ms · cost ${reply.cost_usd:.5f}")


if __name__ == "__main__":
    main()
