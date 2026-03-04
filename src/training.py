import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import TrainConfig
from src.paths import ProjectPaths
from src.logger import get_logger
from src.data_loader import load_sms_dataset
from src.pipeline_factory import build_nb_pipeline, build_lr_pipeline
from src.evaluation import (
    evaluate_model,
    export_error_cases,
    threshold_analysis,
    bootstrap_ci_for_spam,
    save_roc_pr_curves,
    mcnemar_test,
)
from src.reporting import save_report, save_comparison_csv


def train_all(paths: ProjectPaths, cfg: TrainConfig) -> None:
    """
    Train & evaluate multiple models under the SAME split and SAME preprocessing/TF-IDF settings.

    Outputs (results/):
      - report_<model>.txt
      - model_comparison.csv (UPGRADED)
      - error_fp_<model>.csv, error_fn_<model>.csv
      - threshold_analysis_<model>.csv (LR variants)
      - ci_<model>.csv (bootstrap 95% CI for spam F1/Recall)
      - roc_curve_<model>.csv, pr_curve_<model>.csv (LR variants)
      - significance_tests.csv (McNemar pairwise)
    """
    logger = get_logger()
    paths.ensure_dirs()

    df = load_sms_dataset(paths, cfg)
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y
    )

    # models
    models = {
        "nb": (build_nb_pipeline(cfg), cfg.model_nb_name),
        "lr": (build_lr_pipeline(cfg), cfg.model_lr_name),
        "lr_balanced": (build_lr_pipeline(cfg, class_weight="balanced"), cfg.model_lr_balanced_name),
    }

    comparison_rows = []
    preds_cache = {}  # for significance testing

    for model_name, (pipe, filename) in models.items():
        logger.info(f"Training {model_name.upper()} ...")
        pipe.fit(X_train, y_train)

        # evaluate
        res = evaluate_model(pipe, X_test, y_test, cfg, model_name)
        logger.info(
            f"{model_name.upper()} acc={res.accuracy:.4f} macro_f1={res.macro_f1:.4f} "
            f"spam_f1={res.spam_f1:.4f} spam_recall={res.spam_recall:.4f}"
        )

        # predictions for later tests
        y_pred = pipe.predict(X_test)
        preds_cache[model_name] = y_pred

        # bootstrap CI
        ci = bootstrap_ci_for_spam(y_test, y_pred, positive_label=cfg.label_spam, n_boot=1000)
        ci_path = os.path.join(paths.results_dir, f"ci_{model_name}.csv")
        pd.DataFrame([{"model": model_name, **ci}]).to_csv(ci_path, index=False, encoding="utf-8")
        logger.info(
            f"{model_name.upper()} spam_f1={ci['spam_f1_mean']:.4f} "
            f"[{ci['spam_f1_low']:.4f}, {ci['spam_f1_high']:.4f}] (95% CI)"
        )
        logger.info(
            f"{model_name.upper()} spam_recall={ci['spam_recall_mean']:.4f} "
            f"[{ci['spam_recall_low']:.4f}, {ci['spam_recall_high']:.4f}] (95% CI)"
        )

        # save model
        model_path = os.path.join(paths.models_dir, filename)
        joblib.dump(pipe, model_path)
        logger.info(f"Saved model -> {model_path}")

        # report file
        report_path = save_report(paths, cfg, res)
        logger.info(f"Saved report -> {report_path}")

        # error cases
        err_paths = export_error_cases(
            model=pipe,
            X_test=X_test,
            y_test=y_test,
            out_dir=paths.results_dir,
            cfg=cfg,
            model_name=model_name,
            top_k=30
        )
        logger.info(f"Saved FP errors -> {err_paths['fp']}")
        logger.info(f"Saved FN errors -> {err_paths['fn']}")

        # threshold analysis + ROC/PR curve export (only if proba exists)
        th_file = None
        roc_file = None
        pr_file = None
        if hasattr(pipe, "predict_proba"):
            try:
                th_file = threshold_analysis(
                    model=pipe,
                    X_test=X_test,
                    y_test=y_test,
                    out_csv=os.path.join(paths.results_dir, f"threshold_analysis_{model_name}.csv"),
                    cfg=cfg,
                    thresholds=[0.3, 0.5, 0.7]
                )
                logger.info(f"Saved threshold analysis -> {th_file}")
            except Exception as e:
                logger.warning(f"Threshold analysis skipped for {model_name}: {e}")

            try:
                curves = save_roc_pr_curves(
                    model=pipe,
                    X_test=X_test,
                    y_test=y_test,
                    cfg=cfg,
                    out_dir=paths.results_dir,
                    model_name=model_name
                )
                roc_file = curves["roc"]
                pr_file = curves["pr"]
                logger.info(f"Saved ROC curve points -> {roc_file}")
                logger.info(f"Saved PR curve points  -> {pr_file}")
            except Exception as e:
                logger.warning(f"ROC/PR curve export skipped for {model_name}: {e}")

        # upgraded comparison row
        row = {
            "model": model_name,
            "accuracy": round(res.accuracy, 4),
            "macro_f1": round(res.macro_f1, 4),
            "spam_precision": round(res.spam_precision, 4),
            "spam_recall": round(res.spam_recall, 4),
            "spam_f1": round(res.spam_f1, 4),
            "roc_auc": (round(res.roc_auc, 4) if res.roc_auc is not None else ""),
            "pr_auc": (round(res.pr_auc, 4) if res.pr_auc is not None else ""),
            "spam_f1_ci_low": round(ci["spam_f1_low"], 4),
            "spam_f1_ci_high": round(ci["spam_f1_high"], 4),
            "spam_recall_ci_low": round(ci["spam_recall_low"], 4),
            "spam_recall_ci_high": round(ci["spam_recall_high"], 4),
            "threshold_csv": th_file or "",
            "roc_csv": roc_file or "",
            "pr_csv": pr_file or "",
        }
        comparison_rows.append(row)

    # save upgraded comparison table
    comp_path = save_comparison_csv(paths, comparison_rows, filename="model_comparison.csv")
    logger.info(f"Saved comparison CSV -> {comp_path}")

    # ---------------------------
    # Significance testing (McNemar) on same test split
    # pairwise: nb vs lr, lr vs lr_balanced, nb vs lr_balanced
    # ---------------------------
    sig_rows = []
    pairs = [
        ("nb", "lr"),
        ("lr", "lr_balanced"),
        ("nb", "lr_balanced"),
    ]
    for a, b in pairs:
        if a in preds_cache and b in preds_cache:
            res_sig = mcnemar_test(y_test, preds_cache[a], preds_cache[b], positive_label=cfg.label_spam)
            sig_rows.append({
                "model_a": a,
                "model_b": b,
                "b_(a_correct_b_wrong)": res_sig["b"],
                "c_(a_wrong_b_correct)": res_sig["c"],
                "n_discordant": res_sig["n"],
                "p_value": round(res_sig["p_value"], 6),
            })

    sig_path = os.path.join(paths.results_dir, "significance_tests.csv")
    pd.DataFrame(sig_rows).to_csv(sig_path, index=False, encoding="utf-8")
    logger.info(f"Saved significance tests -> {sig_path}")