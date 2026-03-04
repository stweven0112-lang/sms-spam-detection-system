import os
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.utils import resample

from src.config import TrainConfig


@dataclass
class EvalResult:
    model_name: str
    accuracy: float
    report_text: str
    confusion_matrix: np.ndarray

    spam_precision: float
    spam_recall: float
    spam_f1: float
    macro_f1: float

    # Optional (only if predict_proba exists)
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None


def _to_binary(y: pd.Series, positive_label: str) -> np.ndarray:
    y = pd.Series(y).astype(str).str.lower().str.strip()
    return (y.values == str(positive_label).lower()).astype(int)


def evaluate_model(model, X_test, y_test, cfg: TrainConfig, model_name: str) -> EvalResult:
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=[cfg.label_ham, cfg.label_spam])

    report = classification_report(y_test, y_pred, digits=4)

    # spam metrics
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[cfg.label_spam], average=None
    )
    spam_precision = float(p[0]) if len(p) else 0.0
    spam_recall = float(r[0]) if len(r) else 0.0
    spam_f1 = float(f1[0]) if len(f1) else 0.0

    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))

    roc_auc = None
    pr_auc = None

    # ROC/PR AUC need scores
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)
            classes = list(model.classes_)
            if cfg.label_spam in classes:
                spam_idx = classes.index(cfg.label_spam)
                spam_scores = proba[:, spam_idx]
                y_bin = _to_binary(y_test, cfg.label_spam)

                # ROC-AUC (binary)
                roc_auc = float(roc_auc_score(y_bin, spam_scores))
                # PR-AUC (Average Precision)
                pr_auc = float(average_precision_score(y_bin, spam_scores))
        except Exception:
            roc_auc = None
            pr_auc = None

    return EvalResult(
        model_name=model_name,
        accuracy=acc,
        report_text=report,
        confusion_matrix=cm,
        spam_precision=spam_precision,
        spam_recall=spam_recall,
        spam_f1=spam_f1,
        macro_f1=macro_f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
    )


def export_error_cases(
    model,
    X_test,
    y_test,
    out_dir: str,
    cfg: TrainConfig,
    model_name: str,
    top_k: int = 30
) -> Dict[str, str]:
    """
    Export false positives (ham->spam) and false negatives (spam->ham)
    """
    os.makedirs(out_dir, exist_ok=True)
    y_pred = model.predict(X_test)

    df = pd.DataFrame({
        "text": pd.Series(X_test).astype(str).values,
        "y_true": pd.Series(y_test).astype(str).values,
        "y_pred": pd.Series(y_pred).astype(str).values,
    })

    fp = df[(df["y_true"] == cfg.label_ham) & (df["y_pred"] == cfg.label_spam)].head(top_k)
    fn = df[(df["y_true"] == cfg.label_spam) & (df["y_pred"] == cfg.label_ham)].head(top_k)

    fp_path = os.path.join(out_dir, f"error_fp_{model_name}.csv")
    fn_path = os.path.join(out_dir, f"error_fn_{model_name}.csv")

    fp.to_csv(fp_path, index=False, encoding="utf-8")
    fn.to_csv(fn_path, index=False, encoding="utf-8")

    return {"fp": fp_path, "fn": fn_path}


def threshold_analysis(
    model,
    X_test,
    y_test,
    out_csv: str,
    cfg: TrainConfig,
    thresholds: List[float] = None
) -> str:
    """
    Evaluate precision/recall/f1 under different probability thresholds.
    """
    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7]

    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Model has no predict_proba, cannot do threshold analysis.")

    proba = model.predict_proba(X_test)
    classes = list(model.classes_)
    if cfg.label_spam not in classes:
        raise RuntimeError("Spam label not in model.classes_")

    spam_idx = classes.index(cfg.label_spam)
    spam_scores = proba[:, spam_idx]
    y_bin = _to_binary(y_test, cfg.label_spam)

    rows = []
    for th in thresholds:
        y_hat = (spam_scores >= th).astype(int)
        # map back to labels for nicer reporting
        y_pred = np.where(y_hat == 1, cfg.label_spam, cfg.label_ham)

        p, r, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=[cfg.label_spam], average=None
        )
        rows.append({
            "threshold": th,
            "spam_precision": float(p[0]) if len(p) else 0.0,
            "spam_recall": float(r[0]) if len(r) else 0.0,
            "spam_f1": float(f1[0]) if len(f1) else 0.0,
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    return out_csv


def bootstrap_ci_for_spam(
    y_true,
    y_pred,
    positive_label: str,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42
) -> Dict[str, float]:
    """
    Bootstrap 95% CI for spam F1 and spam Recall.
    """
    rng = np.random.RandomState(seed)

    y_true = pd.Series(y_true).astype(str).values
    y_pred = pd.Series(y_pred).astype(str).values

    n = len(y_true)
    f1_list = []
    recall_list = []

    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]

        p, r, f1, _ = precision_recall_fscore_support(
            yt, yp, labels=[positive_label], average=None, zero_division=0
        )
        recall_list.append(float(r[0]) if len(r) else 0.0)
        f1_list.append(float(f1[0]) if len(f1) else 0.0)

    f1_arr = np.array(f1_list)
    r_arr = np.array(recall_list)

    low_q = 100 * (alpha / 2)
    high_q = 100 * (1 - alpha / 2)

    return {
        "spam_f1_mean": float(f1_arr.mean()),
        "spam_f1_low": float(np.percentile(f1_arr, low_q)),
        "spam_f1_high": float(np.percentile(f1_arr, high_q)),
        "spam_recall_mean": float(r_arr.mean()),
        "spam_recall_low": float(np.percentile(r_arr, low_q)),
        "spam_recall_high": float(np.percentile(r_arr, high_q)),
    }


def save_roc_pr_curves(
    model,
    X_test,
    y_test,
    cfg: TrainConfig,
    out_dir: str,
    model_name: str
) -> Dict[str, str]:
    """
    Save ROC curve points + PR curve points into CSV for thesis figures.
    """
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Model has no predict_proba, cannot export ROC/PR curves.")

    proba = model.predict_proba(X_test)
    classes = list(model.classes_)
    if cfg.label_spam not in classes:
        raise RuntimeError("Spam label not in model.classes_")

    spam_scores = proba[:, classes.index(cfg.label_spam)]
    y_bin = _to_binary(y_test, cfg.label_spam)

    fpr, tpr, roc_th = roc_curve(y_bin, spam_scores)
    prec, rec, pr_th = precision_recall_curve(y_bin, spam_scores)

    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_th})
    pr_df = pd.DataFrame({"precision": prec, "recall": rec})

    os.makedirs(out_dir, exist_ok=True)
    roc_path = os.path.join(out_dir, f"roc_curve_{model_name}.csv")
    pr_path = os.path.join(out_dir, f"pr_curve_{model_name}.csv")

    roc_df.to_csv(roc_path, index=False, encoding="utf-8")
    pr_df.to_csv(pr_path, index=False, encoding="utf-8")

    return {"roc": roc_path, "pr": pr_path}


def _binom_two_sided_pvalue(k: int, n: int, p: float = 0.5) -> float:
    """
    Exact two-sided binomial test p-value without SciPy.
    """
    if n == 0:
        return 1.0

    # compute probability mass
    def pmf(i: int) -> float:
        return math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))

    # two-sided: sum probs <= prob(observed)
    p_obs = pmf(k)
    p_val = 0.0
    for i in range(n + 1):
        if pmf(i) <= p_obs + 1e-15:
            p_val += pmf(i)
    return float(min(1.0, p_val))


def mcnemar_test(
    y_true,
    y_pred_a,
    y_pred_b,
    positive_label: str
) -> Dict[str, float]:
    """
    McNemar exact test on paired predictions (A vs B).
    We convert to binary correctness on "positive_label" (spam detection) by comparing
    whether each model is correct on each sample overall (standard McNemar).

    Returns: b, c, n, p_value
      b = A correct, B wrong
      c = A wrong, B correct
      n = b + c
    """
    y_true = pd.Series(y_true).astype(str).values
    a = pd.Series(y_pred_a).astype(str).values
    b = pd.Series(y_pred_b).astype(str).values

    correct_a = (a == y_true)
    correct_b = (b == y_true)

    b_cnt = int(np.sum(correct_a & (~correct_b)))
    c_cnt = int(np.sum((~correct_a) & correct_b))
    n = b_cnt + c_cnt

    # exact binomial with p=0.5, k=min(b,c)
    k = min(b_cnt, c_cnt)
    p_value = _binom_two_sided_pvalue(k=k, n=n, p=0.5)

    return {"b": b_cnt, "c": c_cnt, "n": n, "p_value": p_value}