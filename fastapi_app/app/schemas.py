from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# --- Schemas cho History ---
class StationRecord(BaseModel):
    date: str
    Station_NO: int
    PM2_5: float = Field(..., alias="PM2.5")
    TSP: float
    O3: float
    CO: Optional[float] = 0.0
    NO2: float
    SO2: float
    Temperature: float
    Humidity: float

    class Config:
        populate_by_name = True

# --- Schemas cho Forecast ---
class StationForecastResponse(BaseModel):
    station_id: int
    current_time: str
    forecast_24h: List[float]

class AllStationsForecastResponse(BaseModel):
    current_time: str
    predictions: Dict[str, List[float]]

# --- Schema cho Direct Prediction Request ---
class DirectPredictRequest(BaseModel):
    station_id: int
    # Chuỗi dữ liệu 48 giờ quá khứ
    history_data: List[Dict[str, float]]