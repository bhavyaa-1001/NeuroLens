NeuroLens - Full Stack Setup

Quick start to run the frontend, Node/Express + MongoDB backend, and FastAPI ML service.

Prereqs
- Node 18+
- Python 3.10+
- MongoDB running locally (mongodb://127.0.0.1:27017)

Backend 1: Node/Express + MongoDB (file upload + images API)
1) Create server/.env (optional):
```
MONGO_URI=mongodb://127.0.0.1:27017/neuro_lens
PORT=4000
```
2) Install and start:
```
cd server
npm install
npm run start
```
The server runs at http://localhost:4000
- POST /api/datasets/upload (multipart form-data: dataset=.zip)
- GET  /api/images
- Static files served at /uploads

Backend 2: FastAPI (ML models)
1) Install deps:
```
cd ml
pip install -r requirements.txt
```
2) Start FastAPI:
```
cd server
python app.py
```
The server runs at http://localhost:8000
- POST /run-xgboost
- GET  /health

Note: The Node server extracts uploaded images to server/uploads and mirrors them to ml/data/uploads so FastAPI can read and compute features. If you run FastAPI on another host/port, you can export NODE_PUBLIC_URL env var for image URLs.

Frontend
```
cd frontend
npm install
npm run dev
```
Open http://localhost:5173

Configuration
Frontend env (optional): create frontend/.env with
```
VITE_NODE_API_URL=http://localhost:4000
VITE_FASTAPI_URL=http://localhost:8000
```

Flow
- Upload a dataset .zip in the UI (Workspace page). Node unzips, indexes images in Mongo, and exposes URLs under /uploads.
- Click "Run XGBoost Model". Frontend calls FastAPI /run-xgboost. FastAPI extracts features from ml/data/uploads, loads models if present (ml/models), computes predictions, and returns results. The table will display predicted loss and categories.



