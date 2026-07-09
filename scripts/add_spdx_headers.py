#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Insert SPDX short-form tags into Python files.

Follows SPDX v2.3 guidance for embedded metadata in source files:
https://spdx.github.io/spdx-spec/v2.3/using-SPDX-short-identifiers-in-source-files/

Edit HEADER_LINES below to change the inserted text project-wide.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)

# SPDX tags (Python '#' comments). License must match repo LICENSE (Apache-2.0 here).
HEADER_LINES = [
    "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
    "# SPDX-License-Identifier: Apache-2.0",
]


def iter_py_files(roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            parts = path.parts
            if "__pycache__" in parts or ".venv" in parts:
                continue
            out.append(path)
    return out


def has_spdx_license_identifier(text: str) -> bool:
    return "SPDX-License-Identifier:" in text


def insert_header(content: str) -> str:
    """Return content with SPDX comment block prepended (after shebang if present)."""
    header_block = "\n".join(HEADER_LINES) + "\n\n"
    if content.startswith("#!"):
        idx = content.find("\n")
        if idx == -1:
            return content + "\n" + header_block
        return content[: idx + 1] + header_block + content[idx + 1 :]
    return header_block + content


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Add SPDX headers to Python files.")
    parser.add_argument(
        "--roots",
        nargs="*",
        default=["forecasting", "ad_diffusion", "scripts"],
        type=Path,
        help="Directories to scan (relative to repo root)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with status 1 if any .py file under roots lacks SPDX-License-Identifier",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print paths that would change without writing",
    )
    args = parser.parse_args()

    roots = [(REPO_ROOT / r).resolve() for r in args.roots]
    py_files = iter_py_files(roots)

    missing: list[Path] = []
    changed: list[Path] = []

    for path in py_files:
        text = path.read_text(encoding="utf-8")
        if has_spdx_license_identifier(text):
            continue
        missing.append(path.relative_to(REPO_ROOT))

        new_text = insert_header(text)
        if args.check:
            continue
        if args.dry_run:
            changed.append(path.relative_to(REPO_ROOT))
            logger.info("%s", path.relative_to(REPO_ROOT))
            continue
        path.write_text(new_text, encoding="utf-8", newline="\n")
        changed.append(path.relative_to(REPO_ROOT))

    if args.check:
        if missing:
            logger.error("Missing SPDX-License-Identifier in:")
            for m in missing:
                logger.error("  %s", m)
            return 1
        return 0

    if args.dry_run:
        if changed:
            logger.info("Would update %s file(s)", len(changed))
        else:
            logger.info("Nothing to do")
        return 0

    if changed:
        logger.info("Updated %s file(s)", len(changed))
    else:
        logger.info("Nothing to do (all files already tagged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
