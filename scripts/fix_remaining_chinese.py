from __future__ import annotations

import pathlib
import re
import subprocess

CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

REPLACEMENTS = {
    ">>> 提示:": ">>> Note:",
    "scERso V7 启动 | 任务: 生成式Perturbation预测 | 策略": "scERso V7 started | task: generative perturbation prediction | strategy",
    ",回退Using obs": ", falling back to obs",
    "!!! 警告:": "!!! Warning:",
    "自定义划分列": "Custom split column",
    "正在Load pretrained vectors": "Loading pretrained vectors",
    "个genes的预训练向量,维度": " pretrained gene vectors; dimension",
    "Load pretrained vectors失败": "Failed to load pretrained vectors",
    "当前请求的可视化genes": "Requested visualization gene",
    "个测试Perturbation计算 AUC": " test perturbations",
    "个test setgenes保存至": " test-set genes to",
    "genes {args.heatmap_gene} 不在test set中": "Gene {args.heatmap_gene} is not in the test set",
    "自动选择test set表现最佳genes进行展示": "Automatically selected the best-performing test gene for display",
    ">>> 图saved:": ">>> Figure saved to:",
    ">>> genes报告:": ">>> Gene report:",
}

EXCLUDED = {
    pathlib.Path("scripts/translate_chinese_to_english.py"),
    pathlib.Path("scripts/fix_remaining_chinese.py"),
    pathlib.Path(".github/workflows/scan-chinese.yml"),
    pathlib.Path("CHINESE_SCAN.txt"),
    pathlib.Path("TRANSLATION_LOG.txt"),
}


def tracked_paths() -> list[pathlib.Path]:
    names = subprocess.check_output(["git", "ls-files", "-z"]).decode("utf-8").split("\0")
    return [pathlib.Path(name) for name in names if name]


def main() -> None:
    paths = tracked_paths()
    for path in paths:
        if path in EXCLUDED or not path.exists():
            continue
        try:
            raw = path.read_bytes()
            if b"\x00" in raw:
                continue
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for source, target in REPLACEMENTS.items():
            text = text.replace(source, target)
        path.write_text(text, encoding="utf-8")

    remaining: list[str] = []
    for path in paths:
        if path in EXCLUDED or not path.exists():
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
                remaining.append(f"{path}:{line_no}:{line}")

    pathlib.Path("CHINESE_SCAN.txt").write_text(
        "\n".join(remaining)
        + ("\n" if remaining else "")
        + f"TOTAL_MATCHING_LINES={len(remaining)}\n",
        encoding="utf-8",
    )
    if remaining:
        raise SystemExit(f"{len(remaining)} CJK-containing lines remain")


if __name__ == "__main__":
    main()
