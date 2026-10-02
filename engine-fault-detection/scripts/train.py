"""Train the model offline: python -m scripts.train"""
from app.config import settings
from app.ml.model import FaultDetector

if __name__ == "__main__":
    metrics = FaultDetector(settings.model_path).train(settings.train_samples, settings.random_seed)
    print(f"Accuracy: {metrics['accuracy']:.4f} on {metrics['samples']} samples")
