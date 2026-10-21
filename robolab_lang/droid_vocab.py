#!/usr/bin/env python3
"""Count how often words and phrases occur in the DROID instruction corpus (P4 / P6 ladders).

Input is RoboLab's `analysis/droid_dataset/tasks.jsonl` (31,307 distinct DROID task strings, one
JSON object per line with a "task" field). Counts are over distinct strings, not episodes, so they
measure whether a wording was seen at all and roughly how common it is, not its episode weight.
For episode weights use DROID's `droid_language_annotations.json` (see RoboLab's
analysis/droid_dataset/README.md) with --annotations.

Usage:
    python3 robolab_lang/droid_vocab.py --tasks <RoboLab>/analysis/droid_dataset/tasks.jsonl \
        bin container basket tub "the other" "left bin" "not the"
    python3 robolab_lang/droid_vocab.py --tasks ... --ladder robolab_lang/ladders.txt

A ladder file has one rung per line, `<target>\t<wording>`; the output adds the count per rung so
the rungs can be ordered by corpus frequency before they are frozen as conditions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys


def load_tasks(path: str) -> list[str]:
    out = []
    with open(path) as f:
        for line in f:
            t = json.loads(line).get("task", "").strip().lower()
            if t:
                out.append(t)
    return out


def load_annotations(path: str) -> list[str]:
    d = json.load(open(path))
    out = []
    for v in d.values():
        for k in ("language_instruction1", "language_instruction2", "language_instruction3"):
            t = (v.get(k) or "").strip().lower()
            if t:
                out.append(t)
    return out


def count(corpus: list[str], phrase: str) -> int:
    pat = re.compile(r"\b" + re.escape(phrase.lower()) + r"\b")
    return sum(1 for t in corpus if pat.search(t))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", help="RoboLab analysis/droid_dataset/tasks.jsonl")
    ap.add_argument("--annotations", help="DROID droid_language_annotations.json (episode-level)")
    ap.add_argument("--ladder", help="file of '<target>\\t<wording>' lines")
    ap.add_argument("--examples", type=int, default=0, help="print up to N matching strings per phrase")
    ap.add_argument("phrases", nargs="*")
    args = ap.parse_args(argv)
    if bool(args.tasks) == bool(args.annotations):
        ap.error("give exactly one of --tasks / --annotations")
    corpus = load_tasks(args.tasks) if args.tasks else load_annotations(args.annotations)
    print(f"corpus: {len(corpus)} strings")
    rows = [("", p) for p in args.phrases]
    if args.ladder:
        for line in open(args.ladder):
            line = line.rstrip("\n")
            if line and not line.startswith("#"):
                target, wording = line.split("\t", 1)
                rows.append((target, wording))
    for target, phrase in rows:
        n = count(corpus, phrase)
        print(f"{target:16s} {phrase:40s} {n:7d}  {100 * n / len(corpus):6.2f}%")
        if args.examples:
            pat = re.compile(r"\b" + re.escape(phrase.lower()) + r"\b")
            for t in [t for t in corpus if pat.search(t)][: args.examples]:
                print(f"{'':16s}   {t[:110]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
