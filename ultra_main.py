import os
import time
import sys
import cv2
import numpy as np
import requests
requests.packages.urllib3.disable_warnings()

STREAM_URL = "https://cctvn.freeway.gov.tw/abs2mjpg/bmjpg?camera=20290"

# 顯示縮放（1/2/3，或用 + / - 動態調整）
DISPLAY_SCALE = 2

# 顯示所有偵測類別（包含疑似物體、人、亮光等）
SHOW_ALL_CLASSES = True  # 改為 False 只顯示車輛
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

# Ultralytics YOLO 模型；可改成更大的（效能更好但慢）
# 若您的環境有 YOLOv11，建議用 "yolo11n.pt" 或更大；否則用 yolov8n.pt
MODEL_NAME = os.environ.get("YOLO_MODEL", "yolov8n.pt")


def mjpeg_frames(url: str, timeout: float = 10.0):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    while True:
        try:
            with requests.get(url, stream=True, headers=headers, timeout=timeout, verify=False) as r:
                r.raise_for_status()
                buf = bytearray()
                for chunk in r.iter_content(chunk_size=4096):
                    if chunk:
                        buf.extend(chunk)
                        s = buf.find(b"\xff\xd8")
                        e = buf.find(b"\xff\xd9")
                        if s != -1 and e != -1 and e > s:
                            jpg = buf[s:e+2]
                            buf = buf[e+2:]
                            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if img is not None:
                                yield img
        except Exception as ex:
            print(f"MJPEG stream error: {ex}. Reconnecting in 2s...")
            time.sleep(2.0)


def reduce_glare_bgr(frame: np.ndarray) -> np.ndarray:
    # 輕微減眩處理，保留亮光細節
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # 輕微限制極端高亮（保留更多亮光）
    v = np.clip(v, 0, 250)
    # 輕微 CLAHE 增強對比
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    v = clahe.apply(v)
    hsv = cv2.merge((h, s, v))
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return out


def main():
    global DISPLAY_SCALE  # 允許按鍵修改全域變數
    print(f"Loading Ultralytics YOLO model: {MODEL_NAME}")
    try:
        from ultralytics import YOLO
        import torch
    except Exception as e:
        print("Ultralytics / Torch 未安裝或版本不相容。請先依 README 安裝。")
        print(e)
        sys.exit(1)

    # 選擇裝置：優先使用 GPU（CUDA）
    device = 0 if torch.cuda.is_available() else 'cpu'
    use_half = device != 'cpu'

    model = YOLO(MODEL_NAME)

    prev = time.time()
    fps = 0.0

    print(f"Opening stream: {STREAM_URL}")

    for frame in mjpeg_frames(STREAM_URL):
        # 減眩處理，提升夜間偵測穩定性
        proc = reduce_glare_bgr(frame)

        # 較大輸入尺寸以抓小車；可依 GPU 效能調整（1280/960/640）
        results = model.predict(
            source=proc,
            imgsz=1280,
            conf=0.15,  # 降低信心門檻，偵測更多疑似物體
            iou=0.45,   # 降低 IOU，保留更多重疊框
            device=device,
            half=use_half,
            agnostic_nms=False,
            verbose=False,
        )
        r = results[0]
        names = model.names

        # FPS
        now = time.time()
        dt = now - prev
        prev = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0]) if box.cls is not None else -1
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_name = names.get(cls_id, str(cls_id))
                
                # 依設定過濾或顯示所有類別
                if not SHOW_ALL_CLASSES and cls_name not in VEHICLE_CLASSES:
                    continue
                
                # 根據信心度設定顏色：高信心綠色、中信心黃色、低信心橙色
                if conf >= 0.4:
                    color = (0, 255, 0)  # 綠色 - 高信心
                elif conf >= 0.25:
                    color = (0, 255, 255)  # 黃色 - 中信心
                else:
                    color = (0, 165, 255)  # 橙色 - 低信心（疑似物體）
                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{cls_name} {conf:.2f}", (int(x1), max(0, int(y1)-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        cv2.putText(frame, f"FPS: {fps:.1f} | device: {device}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        disp = frame if DISPLAY_SCALE == 1 else cv2.resize(frame, (frame.shape[1]*DISPLAY_SCALE, frame.shape[0]*DISPLAY_SCALE), interpolation=cv2.INTER_LINEAR)
        win = "Ultralytics YOLO Vehicle Detection (q quit, +/- zoom)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, disp)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('+'):
            DISPLAY_SCALE = min(4, DISPLAY_SCALE + 1)
        elif k == ord('-'):
            DISPLAY_SCALE = max(1, DISPLAY_SCALE - 1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
