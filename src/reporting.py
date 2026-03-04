import os
import csv
from typing import List, Dict

from src.paths import ProjectPaths
from src.evaluation import EvalResult
from src.config import TrainConfig


def save_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def save_report(paths: ProjectPaths, cfg: TrainConfig, res: EvalResult) -> str:
    out = os.path.join(paths.results_dir, f"report_{res.model_name}.txt")

    lines = []
    lines.append(f"Model: {res.model_name}")
    lines.append(f"Accuracy: {res.accuracy:.4f}")
    lines.append(f"Macro-F1: {res.macro_f1:.4f}")
    lines.append(f"Spam Precision: {res.spam_precision:.4f}")
    lines.append(f"Spam Recall: {res.spam_recall:.4f}")
    lines.append(f"Spam F1: {res.spam_f1:.4f}")
    if res.roc_auc is not None:
        lines.append(f"ROC-AUC: {res.roc_auc:.4f}")
    if res.pr_auc is not None:
        lines.append(f"PR-AUC (AP): {res.pr_auc:.4f}")

    lines.append("")
    lines.append("Classification Report:")
    lines.append(res.report_text)
    lines.append("")
    lines.append("Confusion Matrix (rows=true, cols=pred; labels=[ham, spam]):")
    lines.append(str(res.confusion_matrix))

    save_text(out, "\n".join(lines))
    return out


def save_comparison_csv(paths: ProjectPaths, rows: List[Dict], filename: str = "model_comparison.csv") -> str:
    """
    Writes rows to CSV with auto-detected columns (so you can keep adding fields).
    """
    out = os.path.join(paths.results_dir, filename)
    if not rows:
        # create empty file with minimal header
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model"])
        return out

    # union of all keys
    fieldnames = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    return out