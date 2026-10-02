"""Synthetic engine telemetry generator (replace with real sensor data in production)."""
import numpy as np
import pandas as pd

FEATURES = [
    "rpm", "coolant_temp", "oil_pressure", "vibration",
    "exhaust_temp", "fuel_rate", "battery_voltage",
]
FAULTS = ["normal", "overheating", "low_oil_pressure", "bearing_wear", "misfire"]

RECOMMENDATIONS = {
    "normal": "Engine operating within normal parameters.",
    "overheating": "Check coolant level, thermostat, radiator and water pump.",
    "low_oil_pressure": "Inspect oil level, oil pump and filter; stop engine if pressure stays low.",
    "bearing_wear": "Inspect crank/main bearings and schedule an oil analysis.",
    "misfire": "Check spark plugs/injectors, ignition coils and fuel delivery.",
}


def _base(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    rpm = rng.normal(2200, 500, n).clip(700, 5000)
    return {
        "rpm": rpm,
        "coolant_temp": rng.normal(90, 3, n),
        "oil_pressure": 1.5 + rpm / 1500 + rng.normal(0, 0.2, n),
        "vibration": 2 + rpm / 2000 + rng.normal(0, 0.4, n),
        "exhaust_temp": 300 + rpm / 10 + rng.normal(0, 15, n),
        "fuel_rate": 4 + rpm / 400 + rng.normal(0, 0.6, n),
        "battery_voltage": rng.normal(13.8, 0.2, n),
    }


def generate_dataset(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    per_class = n // len(FAULTS)
    frames = []
    for fault in FAULTS:
        d = _base(rng, per_class)
        if fault == "overheating":
            d["coolant_temp"] += rng.normal(25, 6, per_class)
            d["exhaust_temp"] += rng.normal(80, 20, per_class)
        elif fault == "low_oil_pressure":
            d["oil_pressure"] -= rng.normal(1.8, 0.4, per_class)
            d["vibration"] += rng.normal(1.0, 0.4, per_class)
        elif fault == "bearing_wear":
            d["vibration"] += rng.normal(5, 1.2, per_class)
            d["oil_pressure"] -= rng.normal(0.5, 0.2, per_class)
        elif fault == "misfire":
            d["exhaust_temp"] -= rng.normal(70, 20, per_class)
            d["fuel_rate"] += rng.normal(2.5, 0.7, per_class)
            d["vibration"] += rng.normal(2, 0.6, per_class)
            d["battery_voltage"] -= rng.normal(0.4, 0.15, per_class)
        df = pd.DataFrame(d)
        df["oil_pressure"] = df["oil_pressure"].clip(lower=0.1)
        df["fault"] = fault
        frames.append(df)
    return pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
