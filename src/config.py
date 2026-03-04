from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42

    # TF-IDF params
    min_df: int = 2
    ngram_min: int = 1
    ngram_max: int = 2

    # LR params
    lr_max_iter: int = 2000

    label_ham: str = "ham"
    label_spam: str = "spam"

    # output names
    model_nb_name: str = "spam_model_nb.pkl"
    model_lr_name: str = "spam_model_lr.pkl"
    model_lr_balanced_name: str = "spam_model_lr_balanced.pkl"
    report_nb_name: str = "report_nb.txt"
    report_lr_name: str = "report_lr.txt"
    comparison_csv_name: str = "model_comparison.csv"
