# Freeway Vehicle Detection (Ultralytics YOLO + OpenCV)

以台灣高速公路即時監視器 MJPEG 串流為來源，使用 Ultralytics YOLO（v8/v11 皆可）偵測車輛並在畫面上畫框（車、卡車、巴士、機車），支援 GPU 加速與視窗放大。

## 需求
- Windows + Python 3.10 或 3.11（強烈建議；Torch 對過新版本支援度有限）
- 可存取外網（自動下載模型權重）
- NVIDIA 顯卡（選用）：建議安裝最新驅動；Torch GPU 版會自帶 CUDA runtime，無需另裝 CUDA Toolkit。

## 安裝
在工作目錄執行：

```bash
# 建議使用 Python 3.10/3.11 建立虛擬環境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 安裝 OpenCV、Ultralytics（CPU 先行）
pip install -r requirements.txt

# 如要啟用 GPU（NVIDIA）請安裝對應的 Torch 版本（擇一）
# CUDA 12.1（常見）：
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
# 或 CUDA 12.4：
# pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
# 如果沒有 NVIDIA 顯卡或想用 CPU：
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 執行
```bash
python ultra_main.py
```
- 按 `q` 離開；`+/-` 調整視窗放大倍率。
- 若串流偶爾中斷，程式會自動重連。

## 調整建議
- 欲更改串流來源：編輯 `STREAM_URL` 常數。
- 欲切換模型：設環境變數 `YOLO_MODEL=yolo11n.pt`（或 `yolov8n.pt`）。模型愈大偵測愈穩但 FPS 會下降。
- 畫面夜間眩光重：可調整 `reduce_glare_bgr()` 中的參數（如 `clipLimit`）。
- FPS 太低：
  - 啟用 GPU（`device=0`），並使用 `half=True`；
  - 將 `imgsz` 改為 960 或 640；
  - 視窗顯示縮放設小一點（`DISPLAY_SCALE=1`）。

## 已知情況
- MJPEG 來源的 SSL 憑證有時不規範，程式已關閉驗證以提升穩定連線（僅用於公開監視器來源）。
- 若 Torch 在過新 Python 版本（例如 3.12+ 以上）載入失敗，請改用 Python 3.10/3.11。
