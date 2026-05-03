@echo off
echo ============================================
echo   Road Sign Monitor - Starting
echo ============================================

echo [1/2] Starting Prometheus + Grafana...
cd /d C:\Users\arpit\OneDrive\Desktop\road_sign_monitor
docker-compose up -d
timeout /t 5 /nobreak

echo [2/2] Starting FastAPI...
start "FastAPI" cmd /k "C:\Users\arpit\anaconda3\Scripts\activate.bat C:\Users\arpit\anaconda3 && conda activate sperm_gpu && cd /d C:\Users\arpit\OneDrive\Desktop\road_sign_monitor && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

timeout /t 3 /nobreak

echo ============================================
echo   All services running!
echo   FastAPI    : http://localhost:8000/docs
echo   Prometheus : http://localhost:9090
echo   Grafana    : http://localhost:3000
echo ============================================
pause