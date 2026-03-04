import os
import random
import pandas as pd
import joblib

from src.paths import get_project_root, ProjectPaths
from src.config import TrainConfig


def load_models(paths: ProjectPaths, cfg: TrainConfig):
    """
    Load all trained models if present.
    No hard dependency on cfg.model_lr_balanced_name (some configs may not have it).
    """
    candidates = {
        "nb": os.path.join(paths.models_dir, getattr(cfg, "model_nb_name", "spam_model_nb.pkl")),
        "lr": os.path.join(paths.models_dir, getattr(cfg, "model_lr_name", "spam_model_lr.pkl")),

        # ✅ balanced 模型：优先读 config 里有没有字段；没有就用默认文件名
        "lr_balanced": os.path.join(
            paths.models_dir,
            getattr(cfg, "model_lr_balanced_name", "spam_model_lr_balanced.pkl")
        ),
    }

    models = {}
    for name, p in candidates.items():
        if os.path.exists(p):
            models[name] = joblib.load(p)

    if not models:
        raise FileNotFoundError("No models found. Run: python -m src.cli")
    return models


def obfuscate_leetspeak(text: str) -> str:
    mapping = str.maketrans({"a": "@", "e": "3", "i": "1", "o": "0", "s": "$", "t": "7"})
    return text.translate(mapping)


def obfuscate_spacing(text: str) -> str:
    out = []
    for ch in text:
        out.append(ch)
        if ch.isalpha() and random.random() < 0.15:
            out.append(random.choice([" ", ".", "_"]))
    return "".join(out)


def obfuscate_keyword_split(text: str) -> str:
    replacements = {
        "free": "f r e e",
        "prize": "p r i z e",
        "winner": "w i n n e r",
        "claim": "c l a i m",
        "urgent": "u r g e n t",
        "cash": "c a s h",
        "reward": "r e w a r d",
        "click": "c l i c k",
    }
    s = text
    for k, v in replacements.items():
        s = s.replace(k, v)
        s = s.replace(k.upper(), v.upper())
        s = s.replace(k.capitalize(), v.capitalize())
    return s


def make_adversarial(text: str, mode: str) -> str:
    if mode == "leetspeak":
        return obfuscate_leetspeak(text)
    if mode == "spacing":
        return obfuscate_spacing(text)
    if mode == "split_keywords":
        return obfuscate_keyword_split(text)
    return text


def main():
    random.seed(42)
    root = get_project_root(__file__)
    paths = ProjectPaths(root=root)
    cfg = TrainConfig()

    models = load_models(paths, cfg)

    spam_samples = [
        "WINNER!! Claim your FREE prize now!!!",
        "Congratulations! You have won a cash reward. Click the link now.",
        "Free entry in a weekly competition. Text WIN to 87121.",
        "URGENT! Your account has been selected for a prize.",
        "You have been selected for a £1000 reward. Call now to claim.",
        "Claim your free voucher now. Limited time offer!",
        "Congratulations winner! Click here to get your prize.",
        "URGENT! Free tickets available. Reply YES now!",
    ]

    attack_modes = ["original", "leetspeak", "spacing", "split_keywords"]

    # detail rows per sample
    detail_rows = []

    # summary per model + mode
    summary_rows = []

    for model_name, model in models.items():
        # compute baseline (original) recall on spam samples
        orig_preds = []
        for msg in spam_samples:
            pred = model.predict([msg])[0]
            orig_preds.append(pred)
        orig_recall = sum(1 for p in orig_preds if p == cfg.label_spam) / len(spam_samples)

        for mode in attack_modes:
            adv_msgs = []
            preds = []
            probs = []

            for msg in spam_samples:
                adv = msg if mode == "original" else make_adversarial(msg, mode)
                adv_msgs.append(adv)

                pred = model.predict([adv])[0]
                preds.append(pred)

                spam_prob = None
                if hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba([adv])[0]
                        classes = list(model.classes_)
                        if cfg.label_spam in classes:
                            spam_prob = float(proba[classes.index(cfg.label_spam)])
                    except Exception:
                        spam_prob = None
                probs.append(spam_prob)

            adv_recall = sum(1 for p in preds if p == cfg.label_spam) / len(spam_samples)

            drop = (orig_recall - adv_recall)
            drop_pct = (drop / orig_recall * 100.0) if orig_recall > 0 else 0.0

            # evasion rate: among samples originally detected as spam, how many become non-spam after attack
            evaded = 0
            base_detected = 0
            for i in range(len(spam_samples)):
                base_is_spam = (orig_preds[i] == cfg.label_spam)
                adv_is_spam = (preds[i] == cfg.label_spam)
                if base_is_spam:
                    base_detected += 1
                    if not adv_is_spam:
                        evaded += 1
            evasion_rate = (evaded / base_detected) if base_detected > 0 else 0.0

            summary_rows.append({
                "model": model_name,
                "attack_mode": mode,
                "original_spam_recall": round(orig_recall, 4),
                "adversarial_spam_recall": round(adv_recall, 4),
                "recall_drop": round(drop, 4),
                "drop_pct": round(drop_pct, 2),
                "evasion_rate": round(evasion_rate, 4),
                "n_samples": len(spam_samples),
            })

            for i, msg in enumerate(spam_samples):
                detail_rows.append({
                    "model": model_name,
                    "attack_mode": mode,
                    "original_text": msg,
                    "attacked_text": adv_msgs[i],
                    "pred": preds[i],
                    "spam_prob": probs[i],
                })

    # save CSVs
    os.makedirs(paths.results_dir, exist_ok=True)

    detail_path = os.path.join(paths.results_dir, "adversarial_detail.csv")
    pd.DataFrame(detail_rows).to_csv(detail_path, index=False, encoding="utf-8")

    summary_path = os.path.join(paths.results_dir, "adversarial_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8")

    # print summary for screenshot
    print("\n=== Adversarial Summary (per model/mode) ===")
    print(pd.DataFrame(summary_rows).sort_values(["model", "attack_mode"]).to_string(index=False))

    print(f"\n✅ Saved detail:  {detail_path}")
    print(f"✅ Saved summary: {summary_path}")


if __name__ == "__main__":
    main()