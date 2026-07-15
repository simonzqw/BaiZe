from __future__ import annotations

import io
import json
import pathlib
import re
import subprocess
import time
import tokenize
import urllib.parse
import urllib.request

CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
CJK_SEGMENT = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
PROTECTED = re.compile(
    r"(`[^`]*`|\\\([^\n]*?\\\)|\\\[[^\n]*?\\\]|https?://\S+|\{[^{}]*\})"
)
CACHE: dict[str, str] = {}


def normalize_punctuation(text: str) -> str:
    table = str.maketrans(
        {
            "，": ",",
            "。": ".",
            "：": ":",
            "；": ";",
            "！": "!",
            "？": "?",
            "（": "(",
            "）": ")",
            "【": "[",
            "】": "]",
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
            "、": ",",
            "→": "->",
        }
    )
    return text.translate(table)


def google_translate(text: str) -> str:
    text = text.strip()
    if not text or not CJK.search(text):
        return text
    if text in CACHE:
        return CACHE[text]

    params = urllib.parse.urlencode(
        {
            "client": "gtx",
            "sl": "zh-CN",
            "tl": "en",
            "dt": "t",
            "q": text,
        }
    )
    url = "https://translate.googleapis.com/translate_a/single?" + params
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            result = "".join(part[0] for part in payload[0] if part and part[0])
            result = normalize_punctuation(result).strip()
            CACHE[text] = result
            time.sleep(0.05)
            return result
        except Exception as exc:  # pragma: no cover - network retry path
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Translation request failed for: {text!r}") from last_error


def protect(text: str) -> tuple[str, list[str]]:
    values: list[str] = []

    def repl(match: re.Match[str]) -> str:
        values.append(match.group(0))
        return f"ZXQPH{len(values) - 1}QXZ"

    return PROTECTED.sub(repl, text), values


def restore(text: str, values: list[str]) -> str:
    for index, value in enumerate(values):
        text = text.replace(f"ZXQPH{index}QXZ", value)
    return text


def translate_fragment(text: str) -> str:
    if not CJK.search(text):
        return text
    protected, values = protect(text)
    translated = google_translate(protected)
    translated = restore(translated, values)

    if CJK.search(translated):
        translated = CJK_SEGMENT.sub(
            lambda match: google_translate(match.group(0)), translated
        )
    return normalize_punctuation(translated)


def translate_markdown(path: pathlib.Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    output: list[str] = []
    in_fence = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            output.append(line)
            continue
        if in_fence or not CJK.search(line):
            output.append(line)
            continue

        newline = "\n" if line.endswith("\n") else ""
        body = line[:-1] if newline else line
        leading = re.match(r"^\s*", body).group(0)
        translated = translate_fragment(body[len(leading) :])
        output.append(leading + translated + newline)
    path.write_text("".join(output), encoding="utf-8")


def split_string_token(token_text: str) -> tuple[str, str, str] | None:
    match = re.match(r"(?is)^([rubf]*)(\"\"\"|'''|\"|')", token_text)
    if not match:
        return None
    prefix, quote = match.group(1), match.group(2)
    if not token_text.endswith(quote):
        return None
    inner = token_text[len(prefix) + len(quote) : -len(quote)]
    return prefix, quote, inner


def translate_string_token(token_text: str) -> str:
    parsed = split_string_token(token_text)
    if parsed is None:
        return token_text
    prefix, quote, inner = parsed
    if not CJK.search(inner):
        return token_text

    translated_lines: list[str] = []
    for line in inner.splitlines(keepends=True):
        newline = "\n" if line.endswith("\n") else ""
        body = line[:-1] if newline else line
        translated_lines.append(translate_fragment(body) + newline)
    return prefix + quote + "".join(translated_lines) + quote


def translate_python(path: pathlib.Path) -> None:
    source = path.read_text(encoding="utf-8")
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    rewritten: list[tokenize.TokenInfo] = []

    for token in tokens:
        text = token.string
        if token.type == tokenize.COMMENT and CJK.search(text):
            prefix = "#"
            body = text[1:]
            spacing = " " if body.startswith(" ") else ""
            translated = translate_fragment(body.strip())
            text = prefix + spacing + translated
            token = tokenize.TokenInfo(
                token.type, text, token.start, token.end, token.line
            )
        elif token.type == tokenize.STRING and CJK.search(text):
            text = translate_string_token(text)
            token = tokenize.TokenInfo(
                token.type, text, token.start, token.end, token.line
            )
        rewritten.append(token)

    path.write_text(tokenize.untokenize(rewritten), encoding="utf-8")


def translate_text_file(path: pathlib.Path) -> None:
    text = path.read_text(encoding="utf-8")
    lines = []
    for line in text.splitlines(keepends=True):
        if not CJK.search(line):
            lines.append(line)
            continue
        newline = "\n" if line.endswith("\n") else ""
        body = line[:-1] if newline else line
        lines.append(translate_fragment(body) + newline)
    path.write_text("".join(lines), encoding="utf-8")


def tracked_paths() -> list[pathlib.Path]:
    names = subprocess.check_output(["git", "ls-files", "-z"]).decode(
        "utf-8"
    ).split("\0")
    return [pathlib.Path(name) for name in names if name]


def scan_remaining(paths: list[pathlib.Path]) -> list[str]:
    matches: list[str] = []
    for path in paths:
        if not path.exists() or path.name == "CHINESE_SCAN.txt":
            continue
        try:
            raw = path.read_bytes()
            if b"\x00" in raw:
                continue
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            if CJK.search(line):
                matches.append(f"{path}:{line_no}:{line}")
    return matches


def main() -> None:
    old_path = pathlib.Path(
        ".trae/documents/\u5355\u7ec6\u80de\u6270\u52a8\u7279\u5f02\u6027\u5efa\u6a21\u6846\u67b6\u642d\u5efa\u65b9\u6848.md"
    )
    new_path = pathlib.Path(
        ".trae/documents/single_cell_perturbation_specificity_framework_plan.md"
    )
    if old_path.exists():
        new_path.parent.mkdir(parents=True, exist_ok=True)
        old_path.rename(new_path)

    skip = {
        pathlib.Path(".github/workflows/scan-chinese.yml"),
        pathlib.Path("CHINESE_SCAN.txt"),
        pathlib.Path("scripts/translate_chinese_to_english.py"),
    }

    paths = tracked_paths()
    if new_path.exists() and new_path not in paths:
        paths.append(new_path)

    for path in paths:
        if path in skip or not path.exists():
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if not CJK.search(text):
            continue

        if path.suffix == ".py":
            translate_python(path)
        elif path.suffix == ".md":
            translate_markdown(path)
        else:
            translate_text_file(path)

    remaining = scan_remaining(
        [path for path in paths if path not in skip]
    )
    pathlib.Path("CHINESE_SCAN.txt").write_text(
        "\n".join(remaining)
        + ("\n" if remaining else "")
        + f"TOTAL_MATCHING_LINES={len(remaining)}\n",
        encoding="utf-8",
    )
    if remaining:
        raise SystemExit(
            f"Translation finished with {len(remaining)} CJK-containing lines remaining"
        )


if __name__ == "__main__":
    main()
