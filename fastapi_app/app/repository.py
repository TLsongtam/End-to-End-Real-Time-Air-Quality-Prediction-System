from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from pymongo import MongoClient
import pymongo
from app.schemas import StationRecord

class AbstractRepository(ABC):
    @abstractmethod
    def get_station_history(self, station_id: int, limit: int = 48) -> List[dict]:
        """Lấy Lịch sử dữ liệu sensor gần nhất"""
        pass

    @abstractmethod
    def get_latest_forecast(self, station_id: Optional[int] = None) -> dict:
        """Lấy Dự báo 24h mới nhất từ DB"""
        pass


class MongoRepository(AbstractRepository):
    def __init__(self, uri: str, db_name: str):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.history_col = self.db["streaming_history"]
        self.predictions_col = self.db["latest_predictions"]

    def get_station_history(self, station_id: int, limit: int = 48) -> List[dict]:
        # Tìm records theo Station_NO và sắp xếp mới nhất
        cursor = self.history_col.find({
            "$or": [{"Station_NO": station_id}, {"Station_No": station_id}]
        }).sort("_id", pymongo.DESCENDING).limit(limit)

        records = list(cursor)
        for r in records:
            r["_id"] = str(r["_id"])
            if "Station_Nc" in r:
                r["Station_NO"] = r.pop("Station_Nc")
            if "Temperatu" in r:
                r["Temperature"] = r.pop("Temperatu")
        return records[::-1]  # Đảo ngược lại theo thứ tự thời gian tăng dần

    def get_latest_forecast(self, station_id: Optional[int] = None) -> dict:
        # Lấy record dự báo mới nhất lưu trong collection
        latest_doc = self.predictions_col.find_one(sort=[("_id", pymongo.DESCENDING)])
        if not latest_doc:
            return {}
        
        latest_doc["_id"] = str(latest_doc["_id"])
        
        if station_id:
            key_str = f"station_{station_id}"
            predictions = latest_doc.get("predictions", {}).get(key_str, [])
            
            return {
                "station_id": station_id,
                "current_time": latest_doc.get("current_time", ""),
                "forecast_24h": predictions
            }
        return latest_doc