from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    engine_id: Mapped[str] = mapped_column(String(64), index=True)
    rpm: Mapped[float] = mapped_column(Float)
    coolant_temp: Mapped[float] = mapped_column(Float)
    oil_pressure: Mapped[float] = mapped_column(Float)
    vibration: Mapped[float] = mapped_column(Float)
    exhaust_temp: Mapped[float] = mapped_column(Float)
    fuel_rate: Mapped[float] = mapped_column(Float)
    battery_voltage: Mapped[float] = mapped_column(Float)
    predicted_fault: Mapped[str] = mapped_column(String(32), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    is_fault: Mapped[bool] = mapped_column(Boolean)
