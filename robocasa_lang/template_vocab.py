#!/usr/bin/env python3
"""Enumerate the Human300 instruction templates and their vocabulary from the robocasa source (plan §2.3, P4/P6).

Reads the source as text (AST only; nothing from robocasa is imported or executed), so it runs without the
simulator, its assets or the datasets. For every task in TASK_SET_REGISTRY["pretrain300"] it finds the f-string(s)
assigned to ep_meta["lang"] in the task class or its nearest parent, and writes:

  <out>/human300_templates.csv  task, set (atomic_seen / composite_seen / composite_unseen / pretrain_only),
                                n_templates, template (placeholders kept as {expr}), slots
  <out>/human300_vocab.csv      word, source (template / object / fixture), n_tasks (templates containing it),
                                in_target50 (word appears in a target-50 template)

Object words are the OBJ_CATEGORIES keys after object_utils.get_obj_lang's renames; fixture words are the string
constants returned by `nat_lang` properties. Caveat: this is the vocabulary of the *current* code; whether the demos
were collected with the same templates must be checked against the datasets' meta/tasks.jsonl.

Usage: python3 robocasa_lang/template_vocab.py <robocasa checkout> <out dir>
"""
import ast
import csv
import os
import re
import sys
from collections import defaultdict

# object_utils.get_obj_lang renames (robocasa 6f90063, object_utils.py:731)
OBJ_RENAMES = {
    "kettle electric": "electric kettle", "kettle non electric": "kettle", "bread flat": "bread",
    "oil and vinegar bottle": "oil/vinegar bottle", "salt and pepper shaker": "salt/pepper shaker",
    "jug wide opening": "jug",
}
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def parse(path):
    with open(path, encoding="utf-8") as f:
        return ast.parse(f.read(), filename=path)


def is_lang_target(t):
    return (isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and t.slice.value == "lang")


def render(node):
    """Template text of a lang value; None for values we cannot render (e.g. rng.choice(novel_instructions))."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        out = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                out.append(v.value)
            else:
                out.append("{" + ast.unparse(v.value) + "}")
        return "".join(out)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = render(node.left), render(node.right)
        return None if a is None or b is None else a + b
    if isinstance(node, ast.IfExp):
        a, b = render(node.body), render(node.orelse)
        return [x for x in (a, b) if x is not None] or None
    return None


def collect_classes(env_dir):
    classes = {}  # name -> dict(bases, file, langs)
    for root, _, files in os.walk(env_dir):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            for node in ast.walk(parse(path)):
                if not isinstance(node, ast.ClassDef):
                    continue
                langs = []
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Assign) and any(is_lang_target(t) for t in sub.targets):
                        r = render(sub.value)
                        if r:
                            langs.extend(r if isinstance(r, list) else [r])
                bases = [ast.unparse(b).split(".")[-1] for b in node.bases]
                classes[node.name] = dict(bases=bases, file=os.path.relpath(path, env_dir), langs=langs)
    return classes


def resolve(name, classes, seen=()):
    c = classes.get(name)
    if c is None or name in seen:
        return [], None
    if c["langs"]:
        return c["langs"], name
    for b in c["bases"]:
        langs, owner = resolve(b, classes, seen + (name,))
        if langs:
            return langs, owner
    return [], None


def registry_lists(registry_path):
    """TARGET_TASKS split lists and PRETRAINING_TASKS['pretrain300'] from dataset_registry.py (literal lists)."""
    tree = parse(registry_path)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("TARGET_TASKS", "PRETRAINING_TASKS") and isinstance(node.value, ast.Call):
            for kw in node.value.keywords:
                if isinstance(kw.value, ast.List):
                    out[kw.arg] = [e.value for e in kw.value.elts if isinstance(e, ast.Constant)]
    return out


def object_words(objects_path):
    tree = parse(objects_path)
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "OBJ_CATEGORIES":
            cats = [kw.arg for kw in node.value.keywords]
            return sorted({OBJ_RENAMES.get(c.replace("_", " "), c.replace("_", " ")) for c in cats})
    raise RuntimeError("OBJ_CATEGORIES not found")


def fixture_words(fixtures_dir):
    words = set()
    for fn in os.listdir(fixtures_dir):
        if not fn.endswith(".py"):
            continue
        for node in ast.walk(parse(os.path.join(fixtures_dir, fn))):
            if isinstance(node, ast.FunctionDef) and node.name == "nat_lang":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Constant) \
                            and isinstance(sub.value.value, str):
                        words.add(sub.value.value)
    return sorted(words)


def main(robocasa_root, out_dir):
    pkg = os.path.join(robocasa_root, "robocasa")
    classes = collect_classes(os.path.join(pkg, "environments", "kitchen"))
    reg = registry_lists(os.path.join(pkg, "utils", "dataset_registry.py"))
    pretrain = reg["pretrain300"]
    split_of = {t: s for s in ("atomic_seen", "composite_seen", "composite_unseen") for t in reg[s]}

    os.makedirs(out_dir, exist_ok=True)
    word_tasks = defaultdict(set)
    word_target = set()
    missing = []
    with open(os.path.join(out_dir, "human300_templates.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "set", "lang_class", "n_templates", "template", "slots"])
        for task in pretrain + [t for t in split_of if t not in pretrain]:
            langs, owner = resolve(task, classes)
            if not langs:
                missing.append(task)
            for tpl in langs or [""]:
                slots = sorted(set(re.findall(r"\{([^}]*)\}", tpl)))
                w.writerow([task, split_of.get(task, "pretrain_only" if task in pretrain else "?"), owner or "",
                            len(langs), tpl, ";".join(slots)])
                fixed = re.sub(r"\{[^}]*\}", " ", tpl).lower()
                for word in WORD.findall(fixed):
                    word_tasks[word].add(task)
                    if task in split_of:
                        word_target.add(word)

    objs, fxtrs = object_words(os.path.join(pkg, "models", "objects", "kitchen_objects.py")), \
        fixture_words(os.path.join(pkg, "models", "fixtures"))
    with open(os.path.join(out_dir, "human300_vocab.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["word", "source", "n_tasks", "in_target50"])
        for word in sorted(word_tasks, key=lambda x: (-len(word_tasks[x]), x)):
            w.writerow([word, "template", len(word_tasks[word]), word in word_target])
        for o in objs:
            w.writerow([o, "object", "", ""])
        for x in fxtrs:
            w.writerow([x, "fixture", "", ""])

    in_pre = [t for t in pretrain if resolve(t, classes)[0]]
    print(f"pretrain300: {len(pretrain)} tasks, {len(in_pre)} with a template; target tasks not in pretrain300: "
          f"{sorted(t for t in split_of if t not in pretrain)}")
    print(f"template words: {len(word_tasks)}; object categories: {len(objs)}; fixture names: {len(fxtrs)}")
    if missing:
        print(f"no template found ({len(missing)}): {missing}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
