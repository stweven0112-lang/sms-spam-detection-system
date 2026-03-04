from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from src.config import TrainConfig
from src.text_utils import simple_clean


def build_nb_pipeline(cfg: TrainConfig) -> Pipeline:
    vec = TfidfVectorizer(
        preprocessor=simple_clean,
        ngram_range=(cfg.ngram_min, cfg.ngram_max),
        min_df=cfg.min_df,
    )
    clf = MultinomialNB()
    return Pipeline([("tfidf", vec), ("clf", clf)])


def build_lr_pipeline(cfg: TrainConfig, class_weight=None) -> Pipeline:
    vec = TfidfVectorizer(
        preprocessor=simple_clean,
        ngram_range=(cfg.ngram_min, cfg.ngram_max),
        min_df=cfg.min_df,
    )

    clf = LogisticRegression(
        max_iter=cfg.lr_max_iter,
        class_weight=class_weight
    )

    return Pipeline([("tfidf", vec), ("clf", clf)])