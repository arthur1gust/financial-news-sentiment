import argparse
import html
import json
import re
from pathlib import Path

import joblib

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models/day06_inference_pipeline.joblib"
)

LABELS = {
    0: "negative",
    1: "neutral",
    2: "positive"
}


def clean_text(text):
    """Очистка, аналогичная preprocessing Дня 2."""
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def predict_sentiment(texts):
    """Классификация одной или нескольких новостей."""
    if isinstance(texts, str):
        texts = [texts]

    if not isinstance(texts, list) or not texts:
        raise ValueError("Ожидается непустой список строк")

    cleaned = [clean_text(t) for t in texts]

    if any(not t for t in cleaned):
        raise ValueError("Пустой или некорректный текст")

    # Загружаем обученную модель без вызова fit()
    model = joblib.load(MODEL_PATH)

    predictions = model.predict(cleaned)

    probabilities = (
        model.predict_proba(cleaned)
        if hasattr(model, "predict_proba")
        else None
    )

    results = []

    for i, (text, label) in enumerate(
        zip(texts, predictions)
    ):
        item = {
            "text": text,
            "label": int(label),
            "sentiment": LABELS[int(label)]
        }

        if probabilities is not None:
            item["confidence"] = round(
                float(probabilities[i].max()), 4
            )

        results.append(item)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Financial News Sentiment Classification"
    )

    parser.add_argument(
        "--text",
        action="append",
        required=True,
        help="English financial news"
    )

    args = parser.parse_args()

    results = predict_sentiment(args.text)

    print(json.dumps(
        results,
        indent=2,
        ensure_ascii=False
    ))
