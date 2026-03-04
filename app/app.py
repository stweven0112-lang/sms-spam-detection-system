import os
from flask import Flask, render_template, request

from src.paths import get_project_root
from src.config import TrainConfig
from app.services import ModelService

app = Flask(__name__, template_folder="templates", static_folder="static")

root = get_project_root(__file__)
cfg = TrainConfig()

# Load the LR model by default.
model_path = os.path.join(root, "models", cfg.model_lr_name)
service = ModelService(model_path)


@app.route("/", methods=["GET", "POST"])
def index():
    # Set default values first regardless of whether it is a GET or POST request to avoid undefined variables in the template.
    label = None
    spam_prob = None
    prob = None  # Ensure compatibility with older templates that may use the variable name prob as well.
    message = ""
    clean_preview = None

    if request.method == "POST":
        message = request.form.get("message", "")
        if message.strip():
            label, spam_prob, clean_preview = service.predict(message)
            prob = spam_prob

    return render_template(
        "index.html",
        label=label,
        spam_prob=spam_prob,
        prob=prob,  # Ensure compatibility; treat them the same.
        message=message,
        clean_preview=clean_preview
    )


if __name__ == "__main__":
    app.run(debug=True)
