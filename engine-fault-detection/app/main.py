from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_session, init_db
from app.ml.data import FAULTS, FEATURES, RECOMMENDATIONS
from app.ml.model import FaultDetector
from app.models import Prediction
from app.schemas import (
    BatchRequest, PredictionRecord, PredictionResult, SensorReading, TrainResponse,
)

detector = FaultDetector(settings.model_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    detector.load_or_train(settings.train_samples, settings.random_seed)
    yield


app = FastAPI(title="Engine Fault Detection API", version="1.0.0", lifespan=lifespan)


async def _predict_and_store(readings: list[SensorReading], session: AsyncSession) -> list[PredictionResult]:
    results = detector.predict([{f: getattr(r, f) for f in FEATURES} for r in readings])
    out = []
    for r, (fault, conf, probs) in zip(readings, results):
        is_fault = fault != "normal"
        session.add(Prediction(
            engine_id=r.engine_id, predicted_fault=fault, confidence=conf, is_fault=is_fault,
            **{f: getattr(r, f) for f in FEATURES},
        ))
        out.append(PredictionResult(
            engine_id=r.engine_id, predicted_fault=fault, confidence=round(conf, 4),
            is_fault=is_fault, probabilities=probs, recommendation=RECOMMENDATIONS[fault],
        ))
    await session.commit()
    return out


@app.get("/health")
async def health():
    return {"status": "ok", "model_ready": detector.ready}


@app.get("/faults")
async def faults():
    return {"faults": FAULTS, "features": FEATURES}


@app.post("/predict", response_model=PredictionResult)
async def predict(reading: SensorReading, session: AsyncSession = Depends(get_session)):
    return (await _predict_and_store([reading], session))[0]


@app.post("/predict/batch", response_model=list[PredictionResult])
async def predict_batch(body: BatchRequest, session: AsyncSession = Depends(get_session)):
    return await _predict_and_store(body.readings, session)


@app.get("/predictions", response_model=list[PredictionRecord])
async def history(
    engine_id: str | None = None,
    faults_only: bool = False,
    limit: int = Query(50, ge=1, le=500),
    session: AsyncSession = Depends(get_session),
):
    stmt = select(Prediction).order_by(Prediction.id.desc()).limit(limit)
    if engine_id:
        stmt = stmt.where(Prediction.engine_id == engine_id)
    if faults_only:
        stmt = stmt.where(Prediction.is_fault.is_(True))
    return (await session.execute(stmt)).scalars().all()


@app.post("/train", response_model=TrainResponse)
async def retrain():
    try:
        return detector.train(settings.train_samples, settings.random_seed)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc))
