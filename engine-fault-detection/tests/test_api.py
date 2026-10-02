import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_tmp}/test.db"
os.environ["MODEL_PATH"] = f"{_tmp}/model.joblib"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

NORMAL = dict(engine_id="e1", rpm=2200, coolant_temp=90, oil_pressure=3.0,
              vibration=3.1, exhaust_temp=520, fuel_rate=9.5, battery_voltage=13.8)
OVERHEAT = {**NORMAL, "coolant_temp": 118, "exhaust_temp": 600}
BEARING = {**NORMAL, "vibration": 9.5, "oil_pressure": 2.4}


def test_flow():
    with TestClient(app) as c:
        assert c.get("/health").json()["model_ready"] is True
        assert c.post("/predict", json=NORMAL).json()["predicted_fault"] == "normal"
        assert c.post("/predict", json=OVERHEAT).json()["predicted_fault"] == "overheating"
        assert c.post("/predict", json=BEARING).json()["predicted_fault"] == "bearing_wear"
        r = c.post("/predict/batch", json={"readings": [NORMAL, OVERHEAT]})
        assert len(r.json()) == 2
        hist = c.get("/predictions", params={"faults_only": True}).json()
        assert all(h["is_fault"] for h in hist) and len(hist) == 3
        assert c.post("/predict", json={**NORMAL, "rpm": -5}).status_code == 422
        assert c.post("/train").json()["accuracy"] > 0.9
