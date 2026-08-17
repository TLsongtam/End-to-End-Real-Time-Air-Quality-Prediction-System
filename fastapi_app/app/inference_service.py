import os
import torch
import joblib
import numpy as np
from app.config import settings
from app.pytorch_model import build_model  # Code PyTorch model của bạn
import sklearn.preprocessing
import numpy.core.multiarray
import numpy
import numpy.dtypes 

torch.serialization.add_safe_globals([
    sklearn.preprocessing._data.MinMaxScaler,
    numpy.core.multiarray._reconstruct,
    numpy.ndarray,
    numpy.dtype,                  # Thêm sẵn để tránh lỗi kiểu dữ liệu
    numpy.core.multiarray.scalar,  # Thêm sẵn để tránh lỗi số đơn lẻ của Numpy
    numpy.dtypes.Float32DType,
    numpy.dtypes.Float64DType
])

class ModelInferenceService:
    def __init__(self):
        self.models = {}
        self.scalers_X = {}
        self.scalers_y = {}
        self._load_all_models_and_scalers()

    def _load_all_models_and_scalers(self):
        print("⚡ [ML Service] Đang load PyTorch Models và Scalers...")
        for station in range(1, 7):
            n_feat = 7 if station == 2 else 8
            
            # 1. Khởi tạo khung Model
            model_path = os.path.join(settings.MODEL_DIR, f"model_48_24_station{station}.pt")
            model = build_model(input_size=n_feat, config='48_24', station_id=station)
            
            if os.path.exists(model_path):
                # 🛠️ SỬA LỖI TẠI ĐÂY: Load checkpoint dict trước
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                
                # Kiểm tra nếu là dictionary checkpoint thì lấy 'model_state_dict'
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                    
                model.eval()
                self.models[station] = model
            else:
                print(f"⚠️ Warning: Không tìm thấy model weights tại {model_path}")

            # 2. Load Scalers (Ưu tiên lấy từ file .pkl, nếu không có sẽ trích từ checkpoint)
            scaler_x_path = os.path.join(settings.MODEL_DIR, f"scaler_X_station_{station}.pkl")
            scaler_y_path = os.path.join(settings.MODEL_DIR, f"scaler_y_station_{station}.pkl")

            if os.path.exists(scaler_x_path) and os.path.exists(scaler_y_path):
                self.scalers_X[station] = joblib.load(scaler_x_path)
                self.scalers_y[station] = joblib.load(scaler_y_path)
            elif isinstance(checkpoint, dict) and "scaler_X" in checkpoint:
                # Fallback: Lấy trực tiếp scaler đóng gói trong file .pt nếu file .pkl bị thiếu
                self.scalers_X[station] = checkpoint["scaler_X"]
                self.scalers_y[station] = checkpoint["scaler_y"]
            else:
                print(f"⚠️ Warning: Không tìm thấy Scaler cho Station {station}")

        print("✅ [ML Service] Đã sẵn sàng phục vụ Inference!")

    def predict_next_24h(self, station_id: int, input_48h_data: list) -> list:
        """
        input_48h_data: list 48 dicts chứa các chỉ số môi trường
        """
        if station_id not in self.models:
            raise ValueError(f"Station {station_id} không hợp lệ hoặc chưa được load model.")

        features = settings.STATION_2_FEATURES if station_id == 2 else settings.DEFAULT_FEATURES
        
        # Trích xuất dữ liệu theo đúng thứ tự cột
        raw_matrix = []
        for record in input_48h_data:
            row = [record[feat] for feat in features]
            raw_matrix.append(row)
            
        raw_np = np.array(raw_matrix) # Shape (48, num_features)

        # 1. Scale Input
        scaled_x = self.scalers_X[station_id].transform(raw_np)
        
        # 2. PyTorch Forward Pass
        tensor_x = torch.tensor(scaled_x, dtype=torch.float32).unsqueeze(0) # Shape (1, 48, num_features)
        
        with torch.no_grad():
            preds_scaled = self.models[station_id](tensor_x).numpy() # Shape (1, 24)

        # 3. Inverse Scale Output (PM2.5)
        preds_unscaled = self.scalers_y[station_id].inverse_transform(preds_scaled)
        
        return preds_unscaled.flatten().tolist()