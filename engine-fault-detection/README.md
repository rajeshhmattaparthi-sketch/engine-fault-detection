# Engine Fault Detection

FastAPI service that classifies engine faults (overheating, low oil pressure, bearing wear, misfire) from sensor readings using a scikit-learn Random Forest.

**Stack:** FastAPI · async SQLAlchemy 2.0 · Pydantic v2 · scikit-learn · Docker Compose/PostgreSQL · pytest

## Run locally (SQLite)
    pip install -r requirements.txt
    python -m scripts.train        # optional; auto-trains on first start
    uvicorn app.main:app --reload
Docs: http://localhost:8000/docs

## Run with Docker (PostgreSQL)
    docker compose up --build

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | /health | Service + model status |
| GET | /faults | Fault classes and input features |
| POST | /predict | Classify one reading |
| POST | /predict/batch | Classify up to 500 readings |
| GET | /predictions | History (`engine_id`, `faults_only`, `limit`) |
| POST | /train | Retrain the model |

## Example
    curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{"engine_id":"e1","rpm":2200,"coolant_temp":118,"oil_pressure":3.0,"vibration":3.1,"exhaust_temp":600,"fuel_rate":9.5,"battery_voltage":13.8}'

## Tests
    pytest -q

> The training data is synthetic (`app/ml/data.py`). Swap `generate_dataset` for real telemetry (e.g. CSV/CAN-bus logs) for production use.
