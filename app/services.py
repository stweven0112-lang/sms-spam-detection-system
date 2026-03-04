import os
import joblib
from src.text_utils import simple_clean  # must exist for model loading


class ModelService:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Run training first: python -m src.cli"
            )
        self.model = joblib.load(model_path)

    def predict(self, message: str):
        label = self.model.predict([message])[0]
        prob = None
        try:
            proba = self.model.predict_proba([message])[0]
            classes = list(self.model.classes_)
            if "spam" in classes:
                prob = float(proba[classes.index("spam")])
        except Exception:
            prob = None

        preview = simple_clean(message)[:140]
        return label, prob, preview
