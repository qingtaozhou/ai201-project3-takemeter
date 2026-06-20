#!/usr/bin/env python3
"""Local web interface for the space-stock discourse classifier."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs


LABELS = [
    "evidence_analysis",
    "information_update",
    "speculative_hype",
    "risk_skepticism",
]

DATA_PATH = Path("data/labeled_examples.csv")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "fine_tuned_model"))
TOKEN_RE = re.compile(r"\$?[a-zA-Z][a-zA-Z0-9_]*|\d+(?:\.\d+)?%?")
MODEL_WEIGHT_FILES = {
    "model.safetensors",
    "pytorch_model.bin",
    "tf_model.h5",
    "flax_model.msgpack",
}


def load_labels_from_model_config(model_dir: Path = MODEL_DIR) -> list[str]:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return LABELS
    try:
        with config_path.open(encoding="utf-8") as file:
            config = json.load(file)
    except (OSError, json.JSONDecodeError):
        return LABELS

    raw_mapping = config.get("id2label") or {}
    if not raw_mapping:
        return LABELS

    labels = [
        label
        for _, label in sorted(
            ((int(index), label) for index, label in raw_mapping.items()),
            key=lambda item: item[0],
        )
    ]
    return labels or LABELS


LABELS = load_labels_from_model_config()


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


class NaiveBayesTextClassifier:
    backend_name = "CSV Naive Bayes fallback"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.label_counts: Counter[str] = Counter()
        self.token_counts: dict[str, Counter[str]] = {label: Counter() for label in LABELS}
        self.total_tokens: Counter[str] = Counter()
        self.vocabulary: set[str] = set()

    def fit(self, rows: list[dict[str, str]]) -> None:
        for row in rows:
            label = row["label"]
            if label not in LABELS:
                continue
            tokens = tokenize(row["text"])
            self.label_counts[label] += 1
            self.token_counts[label].update(tokens)
            self.total_tokens[label] += len(tokens)
            self.vocabulary.update(tokens)

        missing = [label for label in LABELS if self.label_counts[label] == 0]
        if missing:
            raise ValueError(f"Training data has no examples for labels: {', '.join(missing)}")

    def predict_proba(self, text: str) -> dict[str, float]:
        tokens = tokenize(text)
        total_docs = sum(self.label_counts.values())
        vocab_size = max(len(self.vocabulary), 1)
        log_scores: dict[str, float] = {}

        for label in LABELS:
            prior = math.log(self.label_counts[label] / total_docs)
            denominator = self.total_tokens[label] + self.alpha * vocab_size
            score = prior
            for token in tokens:
                count = self.token_counts[label][token]
                score += math.log((count + self.alpha) / denominator)
            log_scores[label] = score

        max_score = max(log_scores.values())
        exp_scores = {label: math.exp(score - max_score) for label, score in log_scores.items()}
        total = sum(exp_scores.values())
        return {label: exp_scores[label] / total for label in LABELS}

    def predict(self, text: str) -> tuple[str, float, dict[str, float]]:
        probabilities = self.predict_proba(text)
        label = max(probabilities, key=probabilities.get)
        return label, probabilities[label], probabilities


class HuggingFaceTextClassifier:
    backend_name = "fine_tuned_model DistilBERT"

    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install transformers and torch to use fine_tuned_model inference."
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.id2label = self._load_id2label()

    def _load_id2label(self) -> dict[int, str]:
        raw_mapping = getattr(self.model.config, "id2label", None) or {}
        if raw_mapping:
            return {int(index): label for index, label in raw_mapping.items()}
        return {index: label for index, label in enumerate(LABELS)}

    def predict(self, text: str) -> tuple[str, float, dict[str, float]]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs.pop("token_type_ids", None)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probabilities_tensor = self.torch.softmax(logits, dim=-1)

        probabilities = {
            self.id2label.get(index, LABELS[index] if index < len(LABELS) else str(index)): float(probability)
            for index, probability in enumerate(probabilities_tensor)
        }
        label = max(probabilities, key=probabilities.get)
        return label, probabilities[label], probabilities


def load_training_rows(path: Path = DATA_PATH) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find training CSV at {path}")
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"text", "label"}
    missing = required - set(rows[0].keys()) if rows else required
    if missing:
        raise ValueError(f"Training CSV is missing columns: {', '.join(sorted(missing))}")
    return rows


def model_weights_exist(model_dir: Path = MODEL_DIR) -> bool:
    return any((model_dir / filename).exists() for filename in MODEL_WEIGHT_FILES)


def build_classifier() -> object:
    if MODEL_DIR.exists() and model_weights_exist(MODEL_DIR):
        try:
            classifier = HuggingFaceTextClassifier(MODEL_DIR)
            print(f"Loaded classifier backend: {classifier.backend_name}")
            return classifier
        except Exception as exc:
            print(f"Could not load fine-tuned model from {MODEL_DIR}: {exc}")
            print("Falling back to CSV-trained Naive Bayes classifier.")
    elif MODEL_DIR.exists():
        expected = ", ".join(sorted(MODEL_WEIGHT_FILES))
        print(f"Found {MODEL_DIR}, but no model weights. Expected one of: {expected}")
        print("Falling back to CSV-trained Naive Bayes classifier.")

    classifier = NaiveBayesTextClassifier()
    classifier.fit(load_training_rows())
    print(f"Loaded classifier backend: {classifier.backend_name}")
    return classifier


CLASSIFIER = build_classifier()


def render_page(text: str = "", result: dict[str, object] | None = None) -> bytes:
    escaped_text = html.escape(text)
    backend = html.escape(getattr(CLASSIFIER, "backend_name", "classifier"))
    result_html = ""
    if result:
        label = html.escape(str(result["label"]))
        confidence = float(result["confidence"])
        probabilities = result["probabilities"]
        rows = "\n".join(
            f"<tr><td>{html.escape(label_name)}</td><td>{float(prob):.1%}</td></tr>"
            for label_name, prob in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        )
        result_html = f"""
        <section class="result">
          <p class="eyebrow">Prediction</p>
          <h2>{label}</h2>
          <p class="confidence">{confidence:.1%} confidence</p>
          <table>
            <thead><tr><th>Label</th><th>Confidence</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </section>
        """

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Space-Stock Discourse Classifier</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --text: #17202a;
      --muted: #5b6673;
      --line: #cfd7df;
      --accent: #145a8d;
      --accent-strong: #0c4168;
      --surface: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }}
    main {{
      width: min(920px, calc(100% - 32px));
      margin: 40px auto;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 2rem;
      letter-spacing: 0;
    }}
    p {{ color: var(--muted); }}
    .backend {{
      margin-top: 8px;
      color: var(--accent-strong);
      font-weight: 700;
    }}
    form {{
      margin-top: 24px;
      display: grid;
      gap: 12px;
    }}
    label {{
      font-weight: 700;
    }}
    textarea {{
      width: 100%;
      min-height: 180px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      font: inherit;
      color: var(--text);
      background: var(--surface);
    }}
    button {{
      justify-self: start;
      border: 0;
      border-radius: 8px;
      padding: 10px 16px;
      background: var(--accent);
      color: white;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }}
    button:hover {{ background: var(--accent-strong); }}
    .result {{
      margin-top: 28px;
      padding-top: 24px;
      border-top: 1px solid var(--line);
    }}
    .eyebrow {{
      margin: 0;
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--accent-strong);
      font-weight: 800;
    }}
    h2 {{
      margin: 4px 0;
      font-size: 1.6rem;
      letter-spacing: 0;
    }}
    .confidence {{
      margin-top: 0;
      color: var(--text);
      font-weight: 700;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--surface);
      border: 1px solid var(--line);
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
    }}
    th {{ background: #eef3f7; }}
  </style>
</head>
<body>
  <main>
    <h1>Space-Stock Discourse Classifier</h1>
    <p>Paste a public space-stock post to classify its discourse type.</p>
    <p class="backend">Backend: {backend}</p>
    <form method="post" action="/predict">
      <label for="post">Post text</label>
      <textarea id="post" name="text" required>{escaped_text}</textarea>
      <button type="submit">Classify Post</button>
    </form>
    {result_html}
  </main>
</body>
</html>
"""
    return page.encode("utf-8")


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path not in {"/", "/index.html"}:
            self.send_error(404)
            return
        self.respond_html(render_page())

    def do_POST(self) -> None:
        if self.path not in {"/predict", "/api/predict"}:
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        content_type = self.headers.get("Content-Type", "")

        if "application/json" in content_type:
            payload = json.loads(body.decode("utf-8") or "{}")
            text = str(payload.get("text", ""))
        else:
            payload = parse_qs(body.decode("utf-8"))
            text = payload.get("text", [""])[0]

        label, confidence, probabilities = CLASSIFIER.predict(text)
        result = {
            "label": label,
            "confidence": confidence,
            "probabilities": probabilities,
            "backend": getattr(CLASSIFIER, "backend_name", "classifier"),
        }

        if self.path == "/api/predict":
            self.respond_json(result)
        else:
            self.respond_html(render_page(text, result))

    def respond_html(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def respond_json(self, result: dict[str, object]) -> None:
        body = json.dumps(result, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local classifier interface.")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--self-test", action="store_true", help="Print one prediction and exit.")
    args = parser.parse_args()

    if args.self_test:
        sample = "ASTS could be a $100 stock after the next SpaceX launch."
        label, confidence, probabilities = CLASSIFIER.predict(sample)
        print(json.dumps({
            "text": sample,
            "label": label,
            "confidence": confidence,
            "probabilities": probabilities,
            "backend": getattr(CLASSIFIER, "backend_name", "classifier"),
        }, indent=2))
        return

    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"Serving classifier interface at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
