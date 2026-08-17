from fastapi import FastAPI, HTTPException, Depends
from typing import List, Optional
from app.config import settings
from app.schemas import (
    StationRecord, 
    StationForecastResponse, 
    AllStationsForecastResponse, 
    DirectPredictRequest
)
from app.repository import AbstractRepository, MongoRepository
from app.inference_service import ModelInferenceService

app = FastAPI(
    title="Air Quality Real-time Forecasting API",
    description="Backend API phục vụ Hệ thống Dự báo Chất lượng Khai thác PM2.5",
    version="1.0.0"
)

# --- Dependency Injection cho Database ---
def get_repository() -> AbstractRepository:
    return MongoRepository(uri=settings.MONGO_URI, db_name=settings.DATABASE_NAME)

# --- Singleton ML Inference Service ---
ml_service = None

@app.on_event("startup")
def startup_event():
    global ml_service
    ml_service = ModelInferenceService()


# ============================================================
# 4 ROUTES CƠ BẢN
# ============================================================

# 1. Health Check Route
@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "Air Quality Forecasting API",
        "database": "Connected",
        "models_loaded": len(ml_service.models) if ml_service else 0
    }

# 2. Get Stations List
@app.get("/api/v1/stations", tags=["Stations"])
def get_stations():
    return {
        "stations": [
            {"id": 1, "name": "Trạm 1", "features_count": 8},
            {"id": 2, "name": "Trạm 2 (Bỏ CO)", "features_count": 7},
            {"id": 3, "name": "Trạm 3", "features_count": 8},
            {"id": 4, "name": "Trạm 4", "features_count": 8},
            {"id": 5, "name": "Trạm 5", "features_count": 8},
            {"id": 6, "name": "Trạm 6", "features_count": 8},
        ]
    }

# 3. Get 48h Historical Data
@app.get("/api/v1/stations/{station_id}/history", tags=["Air Quality Data"])
def get_station_history(
    station_id: int, 
    limit: int = 48,
    repo: AbstractRepository = Depends(get_repository)
):
    if station_id < 1 or station_id > 6:
        raise HTTPException(status_code=400, detail="station_id phải từ 1 đến 6.")
    
    history = repo.get_station_history(station_id=station_id, limit=limit)
    return {
        "station_id": station_id,
        "count": len(history),
        "data": history
    }

# 4. Get 24h Forecast (Query từ MongoDB)
@app.get("/api/v1/stations/{station_id}/forecast", response_model=StationForecastResponse, tags=["Forecast"])
def get_station_forecast(
    station_id: int,
    repo: AbstractRepository = Depends(get_repository)
):
    if station_id < 1 or station_id > 6:
        raise HTTPException(status_code=400, detail="station_id phải từ 1 đến 6.")
        
    forecast_data = repo.get_latest_forecast(station_id=station_id)
    if not forecast_data:
        raise HTTPException(status_code=444, detail="Chưa có dữ liệu dự báo trong Database.")
        
    return forecast_data


# ============================================================
# EXTENDED ROUTE: Direct Model Inference (Option B - Model Serving)
# ============================================================
@app.post("/api/v1/predict", tags=["Model Serving"])
def predict_directly(request: DirectPredictRequest):
    """
    Truyền vào 48 record dữ liệu lịch sử -> FastAPI gọi PyTorch Model trực tiếp
    để tính toán và trả về 24h dự báo ngay lập tức.
    """
    if len(request.history_data) != 48:
        raise HTTPException(status_code=400, detail="Yêu cầu cung cấp đúng 48 mốc thời gian lịch sử.")
    
    try:
        predictions = ml_service.predict_next_24h(
            station_id=request.station_id, 
            input_48h_data=request.history_data
        )
        return {
            "station_id": request.station_id,
            "forecast_24h_pm25": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))