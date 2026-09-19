#!/usr/bin/env python3
"""Stamp the current version and release date into the citation files.

Usage:  python3 dev_scripts/update_citation.py

The version comes from ``xcal/__init__.py``.  The date comes from the
git tag ``v<version>`` when that tag exists, and from today otherwise.  The
script updates the ``version`` and ``date-released`` fields of CITATION.cff,
and the ``note`` and ``year`` fields of the ``xcal`` BibTeX entry wherever
that entry appears.  Run it after bumping the version and after tagging.
"""
import datetime
import pathlib
import re
import subprocess
import sys

PACKAGE = "xcal"
KEY = "xcal"
FILES = ["CITATION.cff", "README.md", "docs/source/credits.rst"]

ROOT = pathlib.Path(__file__).resolve().parent.parent


def get_version():
    text = (ROOT / PACKAGE / "__init__.py").read_text()
    match = re.search(r"""^__version__\s*=\s*['"]([^'"]+)['"]""", text, re.M)
    if not match:
        sys.exit(f"no __version__ found in {PACKAGE}/__init__.py")
    return match.group(1)


def get_date(version):
    """The date of tag v<version>, or today if that tag does not exist."""
    result = subprocess.run(["git", "log", "-1", "--format=%as", f"v{version}"],
                            cwd=ROOT, capture_output=True, text=True)
    date = result.stdout.strip()
    if result.returncode != 0 or not date:
        date = datetime.date.today().isoformat()
        print(f"tag v{version} not found, using today ({date})")
    return date


def update_cff(text, version, date):
    text = re.sub(r"^version:.*$", f"version: {version}", text, count=1, flags=re.M)
    text = re.sub(r"^date-released:.*$", f"date-released: {date}", text,
                  count=1, flags=re.M)
    return text


def find_entry(text, start):
    """Return (open_brace, close_brace) of the BibTeX entry starting at start."""
    open_brace = text.index("{", start)
    depth = 0
    for i in range(open_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return open_brace, i
    sys.exit("unbalanced braces in a BibTeX entry")


def update_bibtex(text, version, year):
    """Update note and year inside every BibTeX entry whose key is KEY."""
    pattern = re.compile(r"@\s*misc\s*\{\s*" + re.escape(KEY) + r"\s*,", re.I)
    out, pos = [], 0
    for match in pattern.finditer(text):
        open_brace, close_brace = find_entry(text, match.start())
        entry = text[open_brace:close_brace + 1]
        indent = re.search(r"\n(\s*)\w", entry)
        pad = indent.group(1) if indent else "  "
        new = re.sub(r"year[ \t]*=[ \t]*\{?[ \t]*\d{4}[ \t]*\}?", f"year = {year}", entry, count=1)
        if re.search(r"note[ \t]*=", new):
            new = re.sub(r"note[ \t]*=[ \t]*\{[^}]*\}", f"note = {{Version {version}}}",
                         new, count=1)
        else:
            new = new.replace(f"{pad}year = {year}",
                              f"{pad}note = {{Version {version}}},\n{pad}year = {year}", 1)
        out.append(text[pos:open_brace])
        out.append(new)
        pos = close_brace + 1
    out.append(text[pos:])
    return "".join(out)


def main():
    version = get_version()
    date = get_date(version)
    year = date[:4]
    for name in FILES:
        path = ROOT / name
        if not path.exists():
            print(f"skipped {name} (not present)")
            continue
        before = path.read_text()
        after = update_cff(before, version, date) if path.suffix == ".cff" \
            else update_bibtex(before, version, year)
        if after != before:
            path.write_text(after)
            print(f"updated {name}")
        else:
            print(f"unchanged {name}")
    print(f"\n{PACKAGE} {version}, released {date}")


if __name__ == "__main__":
    main()
