"""Run the eval set against the real models:  python -m evals.run  [--min-pass 0.9]

Each case checks the route, words the answer must / must not contain, and the cited source.
Exit code 1 if the pass rate is below --min-pass, so CI can block a bad change.
"""

import argparse
import json
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from app.config import load_settings
from app.factory import build_assistant

CASES = Path(__file__).with_name("cases.jsonl")


def check(case: dict, route: str, text: str) -> list[str]:
    """Return the reasons this case failed (empty list = pass)."""
    lowered = text.lower()
    problems = []
    if case.get("route") and route != case["route"]:
        problems.append(f"route {route}, expected {case['route']}")
    problems += [f"missing '{w}'" for w in case.get("include", []) if w.lower() not in lowered]
    problems += [f"contains '{w}'" for w in case.get("exclude", []) if w.lower() in lowered]
    if case.get("source") and not re.search(rf"\[{re.escape(case['source'])}\]\s*$", text):
        problems.append(f"source is not [{case['source']}]")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-pass", type=float, default=0.9)
    parser.add_argument("--only", help="run only cases whose id contains this text")
    args = parser.parse_args()
    cases = [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]
    cases = [c for c in cases if not args.only or args.only in c["id"]]
    settings = load_settings()

    by_slice: dict[str, list[bool]] = defaultdict(list)
    total_cost = 0.0
    for case in cases:
        # A fresh assistant per case: no cache or pending returns leaking between cases.
        assistant = build_assistant(settings, log_dir=Path(tempfile.mkdtemp()))
        for message in case["turns"]:
            reply = assistant.answer(message, case["customer"])
            total_cost += reply.cost_usd
        problems = check(case, reply.route, reply.text)
        by_slice[case["slice"]].append(not problems)
        mark = "PASS" if not problems else "FAIL"
        print(f"{mark}  {case['id']:24} {'; '.join(problems)}")
        if problems:
            print(f"      answer: {reply.text[:160]}")

    results = [ok for oks in by_slice.values() for ok in oks]
    rate = sum(results) / len(results)
    print("\nby slice: " + " · ".join(f"{s} {sum(v)}/{len(v)}" for s, v in by_slice.items()))
    print(f"pass rate {rate:.0%} ({sum(results)}/{len(results)}) · cost ${total_cost:.4f}")
    sys.exit(0 if rate >= args.min_pass else 1)


if __name__ == "__main__":
    main()
