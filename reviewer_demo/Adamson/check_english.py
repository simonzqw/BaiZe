"""Validate that reviewer-facing text and runtime logs contain no Han characters."""
import argparse
from pathlib import Path
import re
import sys

HAN = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")


def main():
    parser = argparse.ArgumentParser(description="Check English-only reviewer-facing files")
    parser.add_argument("paths", nargs="*", help="Files to check (UTF-8)")
    options = parser.parse_args()
    defaults = [
        "README.md", "ENVIRONMENT.md", "requirements.txt", "run_evaluation.sh",
        "run_training_smoke.sh", "verify_package.sh", "english_console.py",
        "check_english.py", "INSTALL.md", "download_checkpoint.sh",
        "PUBLICATION_STATUS.md", "SOURCE_PROVENANCE.md"
    ]
    paths = options.paths or defaults
    failures = 0
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            print(f"ERROR: File does not exist: {p}", file=sys.stderr)
            failures += 1
            continue
        for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if HAN.search(line):
                print(f"ERROR: Non-English text at {p}:{lineno}", file=sys.stderr)
                failures += 1
    if failures:
        print(f"FAIL: {failures} non-English entries detected", file=sys.stderr)
        sys.exit(2)
    print(f"PASS: English-only text check ({len(paths)} file(s))")


if __name__ == "__main__":
    main()
