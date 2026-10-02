from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.data import FEATURES, generate_dataset


class FaultDetector:
    def __init__(self, path: str):
        self.path = Path(path)
        self.pipeline: Pipeline | None = None

    @property
    def ready(self) -> bool:
        return self.pipeline is not None

    def train(self, n_samples: int = 6000, seed: int = 42) -> dict:
        df = generate_dataset(n_samples, seed)
        X_train, X_test, y_train, y_test = train_test_split(
            df[FEATURES], df["fault"], test_size=0.2, stratify=df["fault"], random_state=seed
        )
        self.pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)),
        ])
        self.pipeline.fit(X_train, y_train)
        pred = self.pipeline.predict(X_test)
        report = classification_report(y_test, pred, output_dict=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, self.path)
        return {"accuracy": float(report["accuracy"]), "samples": len(df), "report": report}

    def load_or_train(self, n_samples: int = 6000, seed: int = 42) -> None:
        if self.path.exists():
            self.pipeline = joblib.load(self.path)
        else:
            self.train(n_samples, seed)

    def predict(self, rows: list[dict]) -> list[tuple[str, float, dict[str, float]]]:
        if not self.pipeline:
            raise RuntimeError("Model not loaded")
        X = pd.DataFrame(rows)[FEATURES]
        proba = self.pipeline.predict_proba(X)
        classes = list(self.pipeline.classes_)
        out = []
        for p in proba:
            idx = int(p.argmax())
            out.append((classes[idx], float(p[idx]), {c: round(float(v), 4) for c, v in zip(classes, p)}))
        return out
