"""English-only console messages for the BaiZe Adamson reviewer demonstration.

This module changes *displayed text only*. It does not modify model definitions,
weights, preprocessing, splits, sampling, or metric calculations.

Legacy source contains non-English log literals. Known phrases are translated,
and any unrecognized CJK message triggers an explicit English error rather than
silently hiding scientific diagnostics.
"""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import re
import sys

# Ordered translations: longer, specific phrases appear before generic phrases.
PHRASES = (
    ("perturb_parse_mode=raw: \u4fdd\u7559\u5b57\u7b26\u4e32\uff0c\u4f46\u81ea\u52a8\u8bc6\u522b Adamson UPR \u548c plus-separated \u591a\u57fa\u56e0\u6270\u52a8\u3002",
     "perturb_parse_mode=raw: preserving labels and detecting Adamson UPR and plus-separated multi-gene perturbations."),
    ("\u68c0\u6d4b\u5230\u5242\u91cf\u4fe1\u606f\uff0c\u6b63\u5728\u5904\u7406 Dose \u7279\u5f81...",
     "Dose information detected; processing dose features..."),
    ("\u6b63\u5728\u8ba1\u7b97 control context \u57fa\u7ebf\u8868\u8fbe\u8c31...",
     "Computing control-context baseline expression profiles..."),
    ("\u6b63\u5728\u8ba1\u7b97 control context \u57fa\u7ebf\u8868\u8fbe\u8c31",
     "Computing control-context baseline expression profiles"),
    ("\u6b63\u5728\u8ba1\u7b97 control context \u57fa\u7ebf\u8868\u8fbe\u8c31...",
     "Computing baseline expression profiles for control contexts..."),
    ("\u91c7\u7528\u81ea\u5b9a\u4e49\u5212\u5206\u7b56\u7565", "Using predefined data split"),
    ("\u4f7f\u7528\u81ea\u5b9a\u4e49\u5212\u5206\u7b56\u7565", "Using predefined data split"),
    ("\u91c7\u7528\u81ea\u5b9a\u4e49\u5212\u5206", "Using predefined data split"),
    ("\u6570\u636e\u52a0\u8f7d\u5b8c\u6210", "Data loaded"),
    ("\u6b63\u5728\u52a0\u8f7d\u6570\u636e", "Loading data"),
    ("\u52a0\u8f7d\u6570\u636e", "Loading data"),
    ("\u5212\u5206\u7ed3\u679c", "Split sizes"),
    ("\u5206\u5272\u7ed3\u679c", "Split sizes"),
    ("\u5df2\u5199\u5165", "Prepared metadata fields"),
    ("\u6307\u6807\u5df2\u4fdd\u5b58\u5230", "Evaluation metrics saved to"),
    ("\u8bc4\u4f30\u7ed3\u679c\u5df2\u4fdd\u5b58\u5230", "Evaluation metrics saved to"),
    ("\u6a21\u578b\u5df2\u4fdd\u5b58\u5230", "Checkpoint saved to"),
    ("\u4fdd\u5b58\u6a21\u578b\u5230", "Saving checkpoint to"),
    ("\u8bad\u7ec3\u5b8c\u6210", "Training completed"),
    ("\u8bad\u7ec3\u7ed3\u675f", "Training finished"),
    ("\u5f00\u59cb\u8bad\u7ec3", "Starting training"),
    ("\u9a8c\u8bc1\u5b8c\u6210", "Validation completed"),
    ("\u6d4b\u8bd5\u5b8c\u6210", "Test completed"),
    ("\u6b63\u5728\u8bc4\u4f30", "Evaluating"),
    ("\u672a\u627e\u5230", "Not found"),
    ("\u6570\u636e\u5212\u5206", "Data split"),
    ("\u68c0\u6d4b\u5230", "Detected"),
    ("\u57fa\u56e0\u6270\u52a8", "gene perturbations"),
    ("\u591a\u57fa\u56e0\u6270\u52a8", "multi-gene perturbations"),
    ("\u5355\u57fa\u56e0\u6270\u52a8", "single-gene perturbations"),
    ("\u5bf9\u7167\u7ec6\u80de", "control cells"),
    ("\u6270\u52a8\u7ec6\u80de", "perturbed cells"),
    ("\u7ec6\u80de\u6570\u91cf", "cell count"),
    ("\u57fa\u56e0\u6570\u91cf", "gene count"),
    ("\u7ec6\u80de", "cells"),
    ("\u57fa\u56e0", "genes"),
    ("\u8bad\u7ec3\u96c6", "training partition"),
    ("\u9a8c\u8bc1\u96c6", "validation partition"),
    ("\u6d4b\u8bd5\u96c6", "test partition"),
    ("\u968f\u673a\u79cd\u5b50", "random seed"),
    ("\u9519\u8bef", "ERROR"),
    ("\u8b66\u544a", "WARNING"),
    ("\u6ce8\u610f", "NOTICE"),
    ("\u5931\u8d25", "FAILED"),
    ("\u6210\u529f", "SUCCESS"),
    ("\u5b8c\u6210", "completed"),
    ("\u6b63\u5728", "Processing "),
    ("\u5904\u7406", "processing"),
    ("\u4fe1\u606f", "information"),
    ("\u8def\u5f84", "path"),
    ("\u7ed3\u679c", "results"),
    ("\u6570\u91cf", "count"),
    ("\u672a\u89c1", "held-out"),
    ("\u4fdd\u5b58", "saved"),
    ("\u52a0\u8f7d", "loading"),
    ("\u4f7f\u7528", "using"),
    ("\u91c7\u7528", "using"),
    ("\u6ca1\u6709", "no"),
    ("\u7f3a\u5c11", "missing"),
    ("\u4e3a\u7a7a", "is empty"),
    ("\u662f", "is"),
    ("\u548c", "and"),
    ("\u4e0e", "and"),
)

HAN = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
PUNCT = str.maketrans({"\uff1a": ":", "\uff0c": ",", "\u3002": ".", "\uff08": "(", "\uff09": ")", "\u3001": ",", "\uff1b": ";", "\uff01": "!", "\uff1f": "?", "\u3010": "[", "\u3011": "]", "\uff0d": "-"})


def english_text(text):
    """Translate recognized legacy log phrases, refusing unknown CJK content."""
    for old, new in PHRASES:
        text = text.replace(old, new)
    text = text.translate(PUNCT)
    if HAN.search(text):
        escaped = text.encode("ascii", errors="backslashreplace").decode("ascii")
        raise RuntimeError(
            "English-only log check failed: an untranslated source message "
            "was encountered. Add an English translation in english_console.py. "
            "Escaped source text: " + escaped[:700]
        )
    return text


class EnglishLogStream(io.TextIOBase):
    """Preserve line/progress-bar boundaries while translating source logs."""
    def __init__(self, target):
        self.target = target
        self.pending = ""

    @property
    def encoding(self):
        return getattr(self.target, "encoding", "utf-8")

    def writable(self):
        return True

    def isatty(self):
        return self.target.isatty()

    def fileno(self):
        return self.target.fileno()

    def write(self, message):
        if not isinstance(message, str):
            message = str(message)
        self.pending += message
        while True:
            n = [pos for pos in (self.pending.find("\n"), self.pending.find("\r")) if pos >= 0]
            if not n:
                break
            end = min(n) + 1
            segment = self.pending[:end]
            self.pending = self.pending[end:]
            self.target.write(english_text(segment))
        return len(message)

    def flush(self):
        if self.pending:
            self.target.write(english_text(self.pending))
            self.pending = ""
        self.target.flush()


@contextmanager
def english_output():
    out = EnglishLogStream(sys.stdout)
    err = EnglishLogStream(sys.stderr)
    try:
        with redirect_stdout(out), redirect_stderr(err):
            yield
    finally:
        out.flush()
        err.flush()


if __name__ == "__main__":
    examples = [
        ">>> Loading data: demo.h5ad\n",
        ">>> Data loaded: 2020 cells, 5060 genes\n",
    ]
    for example in examples:
        print(english_text(example), end="")
    print("PASS: English console translator imported successfully")
