# Real-Time Air Quality Forecasting System

This project demonstrates a **real-time air quality streaming and PM2.5 forecasting pipeline** using Kafka, Spark, PyTorch, MongoDB Atlas, FastAPI, and Streamlit.

The system simulates air quality data from **6 stations in Ho Chi Minh City**, streams it through Kafka, processes it with Spark Structured Streaming, generates **24-hour PM2.5 forecasts** using BiLSTM-MLAM models, stores results in MongoDB Atlas, and visualizes them through a Streamlit dashboard.

---

## Architecture

1. **Data Producer**
   `spark_kafka/producer.py` reads simulated air quality data and sends records to the `air_quality` Kafka topic.

2. **Kafka**
   Acts as the real-time messaging layer between the producer and Spark.

3. **Spark Streaming**
   `spark_kafka/spark_consumer_mongodb.py` consumes Kafka data, maintains a **48-hour history window**, and runs PyTorch inference to predict the next **24 hours of PM2.5**.

4. **MongoDB Atlas**
   Stores actual streaming data in `streaming_history` and forecasts in `latest_predictions`.

5. **FastAPI & Streamlit**
   FastAPI provides the backend API for MongoDB data, while Streamlit displays real-time measurements and forecasts.

```text
CSV → Kafka → Spark Streaming → PyTorch → MongoDB Atlas → FastAPI → Streamlit
```

---

## Models & Forecasting

* **Model:** BiLSTM-MLAM (Bidirectional LSTM with Multi-Scale Local Attention Mechanism)
* **Input:** 48 hours of historical air quality data
* **Output:** 24-hour PM2.5 forecast
* **Stations:** 6
* **Framework:** PyTorch
* **Scalers:** Scikit-Learn / Joblib

Each station has its own trained model and feature scalers.

---

## Directory Structure

```text
.
├── Docker/
│   ├── Dockerfile.fastapi
│   ├── Dockerfile.streamlit
│   └── docker-compose.yml
├── app/
│   ├── app_dashboard_api.py
│   ├── config.py
│   ├── inference_service.py
│   ├── main.py
│   ├── pytorch_model.py
│   ├── repository.py
│   └── schemas.py
├── data/
│   ├── latest_predictions.json
│   ├── simulation_stream.csv
│   └── streaming_history.csv
├── img/
│   ├── API.png
│   ├── Pipeline.png
│   └── Streamlit.png
├── models/
│   ├── model_48_24_station1.pt
│   ├── model_48_24_station2.pt
│   ├── model_48_24_station3.pt
│   ├── model_48_24_station4.pt
│   ├── model_48_24_station5.pt
│   ├── model_48_24_station6.pt
│   ├── scaler_X_station_1.pkl
│   ├── ...
│   └── scaler_y_station_6.pkl
├── spark_kafka/
│   ├── producer.py
│   └── spark_consumer_mongodb.py
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

### Directory Description

* **`Docker/`**: Dockerfiles and Docker Compose configuration for FastAPI and Streamlit.
* **`app/`**: FastAPI backend, MongoDB repository, inference service, PyTorch model loader, schemas, and Streamlit dashboard.
* **`data/`**: Simulated streaming data and local prediction/history files.
* **`img/`**: Architecture and application screenshots.
* **`models/`**: Pre-trained PyTorch models and Scikit-Learn scalers for 6 stations.
* **`spark_kafka/`**: Kafka producer and Spark Structured Streaming consumer.
* **`.env`**: Runtime configuration such as MongoDB, Kafka, and model paths. This file should **not** be committed to Git.
* **`requirements.txt`**: Python dependencies.

---

## Setup & Run

### 1. Clone the repository

```bash
git clone https://github.com/TLsongtam/Real-time-Air-Quality-Forecasting-System-using-PyTorch-Spark-Streaming.git
cd Real-time-Air-Quality-Forecasting-System-using-PyTorch-Spark-Streaming
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Create `.env` in the project root:

```env
MONGO_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/
DATABASE_NAME=air_quality_db
MODEL_DIR=/path/to/models

KAFKA_BROKER=localhost:9092
TOPIC_NAME=air_quality
DATA_STREAM_FILE=data/simulation_stream.csv
```

### 3. Start Kafka

Make sure Kafka is running on:

```text
localhost:9092
```

### 4. Start Spark Streaming

```bash
python spark_kafka/spark_consumer_mongodb.py
```

### 5. Start the Producer

In another terminal:

```bash
python spark_kafka/producer.py
```

### 6. Start FastAPI

```bash
uvicorn app.main:app --reload
```

API documentation:

```text
http://localhost:8000/docs
```

### 7. Start Streamlit

```bash
streamlit run app/app_dashboard_api.py
```

Dashboard:

```text
http://localhost:8501
```

---

## MongoDB

The application uses MongoDB Atlas with:

```text
air_quality_db
├── streaming_history
└── latest_predictions
```

`streaming_history` stores real-time sensor measurements, while `latest_predictions` stores the latest 24-hour PM2.5 forecasts.

---

## Docker

To run the FastAPI and Streamlit services with Docker:

```bash
docker compose -f Docker/docker-compose.yml up --build
```

Stop services:

```bash
docker compose -f Docker/docker-compose.yml down
```

---

## Conclusion

This project demonstrates an end-to-end **real-time streaming ML pipeline** combining Kafka, Spark Structured Streaming, PyTorch, MongoDB Atlas, FastAPI, and Streamlit for real-time air quality monitoring and 24-hour PM2.5 forecasting.
