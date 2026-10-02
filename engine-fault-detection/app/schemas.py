from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class SensorReading(BaseModel):
    engine_id: str = Field(default="engine-001", max_length=64)
    rpm: float = Field(ge=0, le=8000, description="Engine speed (rev/min)")
    coolant_temp: float = Field(ge=-40, le=200, description="Coolant temperature (°C)")
    oil_pressure: float = Field(ge=0, le=12, description="Oil pressure (bar)")
    vibration: float = Field(ge=0, le=50, description="Vibration RMS (mm/s)")
    exhaust_temp: float = Field(ge=0, le=1200, description="Exhaust gas temperature (°C)")
    fuel_rate: float = Field(ge=0, le=100, description="Fuel consumption (L/h)")
    battery_voltage: float = Field(ge=0, le=30, description="Battery voltage (V)")


class PredictionResult(BaseModel):
    engine_id: str
    predicted_fault: str
    confidence: float
    is_fault: bool
    probabilities: dict[str, float]
    recommendation: str


class BatchRequest(BaseModel):
    readings: list[SensorReading] = Field(min_length=1, max_length=500)


class PredictionRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    engine_id: str
    predicted_fault: str
    confidence: float
    is_fault: bool


class TrainResponse(BaseModel):
    accuracy: float
    samples: int
    report: dict
