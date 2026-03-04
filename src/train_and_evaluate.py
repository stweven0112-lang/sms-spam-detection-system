import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.text_utils import simple_clean
from src.evaluation import evaluate_model, export_error_cases, threshold_analysis
from src.config import TrainConfig


def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_dataset():
    data_path = os.path.join(project_root(), "data", "spam.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path, encoding="latin-1")
    df = df.iloc[:, :2].copy()
    df.columns = ["label", "text"]
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["text"] = df["text"].astype(str)
    return df


def build_pipeline(model_name: str, cfg: TrainConfig):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB

    if model_name == "lr":
        clf = LogisticRegression(max_iter=cfg.lr_max_iter)
    elif model_name == "nb":
        clf = MultinomialNB()
    else:
        raise ValueError("model_name must be 'lr' or 'nb'")

    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=simple_clean,
            ngram_range=(cfg.ngram_min, cfg.ngram_max),
            min_df=cfg.min_df
        )),
        ("clf", clf)
    ])


def main():
    cfg = TrainConfig()
    root = project_root()
    models_dir = os.path.join(root, "models")
    results_dir = os.path.join(root, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    df = load_dataset()
    X, y = df["text"], df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    comparison_rows = []

    for model_name in ["nb", "lr"]:
        print(f"\n=== Training {model_name.upper()} ===")

        model = build_pipeline(model_name, cfg)
        model.fit(X_train, y_train)

        # Evaluate (same output format as training.py)
        res = evaluate_model(model, X_test, y_test, cfg, model_name)

        # Save model
        model_path = os.path.join(models_dir, f"spam_model_{model_name}.pkl")
        joblib.dump(model, model_path)

        # Save report
        report_path = os.path.join(results_dir, f"report_{model_name}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(res.report_text)
            f.write("\n\nConfusion Matrix (rows=true, cols=pred; labels=[ham, spam]):\n")
            f.write(str(res.confusion_matrix))

        # Export FP/FN
        export_error_cases(
            model=model,
            X_test=X_test,
            y_test=y_test,
            out_dir=results_dir,
            cfg=cfg,
            model_name=model_name,
            top_k=30
        )

        # Threshold analysis (if possible)
        if hasattr(model, "predict_proba"):
            try:
                threshold_analysis(
                    model=model,
                    X_test=X_test,
                    y_test=y_test,
                    out_csv=os.path.join(results_dir, f"threshold_analysis_{model_name}.csv"),
                    cfg=cfg,
                    thresholds=[0.3, 0.5, 0.7]
                )
            except Exception:
                pass

        comparison_rows.append({
            "model": model_name,
            "accuracy": round(res.accuracy, 4),
            "spam_precision": round(float(res.spam_precision or 0.0), 4),
            "spam_recall": round(float(res.spam_recall or 0.0), 4),
            "spam_f1": round(float(res.spam_f1 or 0.0), 4),
        })

        print(f"Saved model -> {model_path}")
        print(f"Saved report -> {report_path}")

    # Save comparison CSV
    comp_df = pd.DataFrame(comparison_rows).sort_values("model")
    comp_path = os.path.join(results_dir, "model_comparison.csv")
    comp_df.to_csv(comp_path, index=False, encoding="utf-8")
    print(f"\n✅ Saved comparison CSV -> {comp_path}")
    print(comp_df)


if __name__ == "__main__":
    main()