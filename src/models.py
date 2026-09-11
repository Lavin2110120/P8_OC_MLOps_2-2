from datetime import datetime, timezone
from sqlalchemy import Integer, Float, String, JSON, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from src.database import Base

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    inputs: Mapped[dict] = mapped_column(JSON, nullable=False)
    prediction: Mapped[int] = mapped_column(Integer, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    engine: Mapped[str] = mapped_column(String(50), default="onnxruntime")
    status: Mapped[str] = mapped_column(String(20), default="success")