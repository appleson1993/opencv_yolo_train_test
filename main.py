import os
import time
import sys
import cv2
import numpy as np
import urllib.request
import requests
requests.packages.urllib3.disable_warnings()  # 抑制 verify=False 警告

STREAM_URL = "https://cctvn.freeway.gov.tw/abs2mjpg/bmjpg?camera=20290"

# 目標類別：僅標示車輛相關（coco.names 使用 motorbike）
VEHICLE_CLASSES = {"car", "truck", "bus", "motorbike"}

# 顯示放大倍率（只影響顯示，不影響偵測）
DISPLAY_SCALE = 2  # 1、2、3 可調
ZOOM_KEYS = True   # 允許用 + / - 動態放大縮小

# 第二次偵測的 ROI（相對比例 x, y, w, h），針對遠處小車加強
SECOND_PASS_ROI = (0.0, 0.0, 1.0, 0.6)

# 將權重存到 ASCII 路徑以避免部分庫對 Unicode 路徑不友善
import tempfile
WEIGHTS_DIR = os.path.join(tempfile.gettempdir(), "cv_freeway_weights")
CFG_PATH = os.path.join(WEIGHTS_DIR, "yolo.cfg")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "yolo.weights")
NAMES_PATH = os.path.join(WEIGHTS_DIR, "coco.names")

# 預設使用 YOLOv4-tiny（通常比 v3-tiny 準）
CFG_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
WEIGHTS_URL = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
NAMES_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"


def ensure_file(path: str, url: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"Downloading {os.path.basename(path)}...")
        urllib.request.urlretrieve(url, path)


def load_yolo():
    ensure_file(CFG_PATH, CFG_URL)
    ensure_file(WEIGHTS_PATH, WEIGHTS_URL)
    ensure_file(NAMES_PATH, NAMES_URL)

    with open(NAMES_PATH, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]

    net = cv2.dnn.readNetFromDarknet(CFG_PATH, WEIGHTS_PATH)
    # 使用 CPU（若有 OpenCV CUDA 可切換）
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    layer_names = net.getLayerNames()
    out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, out_layers, classes


def mjpeg_frames(url: str, timeout: float = 10.0):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    while True:
        try:
            with requests.get(url, stream=True, headers=headers, timeout=timeout, verify=False) as r:
                r.raise_for_status()
                buffer = bytearray()
                for chunk in r.iter_content(chunk_size=4096):
                    if chunk:
                        buffer.extend(chunk)
                        # 尋找 JPEG 影像的頭尾標記
                        start = buffer.find(b"\xff\xd8")
                        end = buffer.find(b"\xff\xd9")
                        if start != -1 and end != -1 and end > start:
                            jpg = buffer[start:end + 2]
                            buffer = buffer[end + 2:]
                            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if img is not None:
                                yield img
        except Exception as e:
            print(f"MJPEG stream error: {e}. Reconnecting in 2s...")
            time.sleep(2.0)


def draw_label(frame, x1, y1, x2, y2, label, color=(0, 255, 0)):
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    cv2.putText(
        frame,
        label,
        (int(x1), max(0, int(y1) - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def detect_vehicles(frame, net, out_layers, classes, conf_thresh=0.25, nms_thresh=0.45, blob_size=608):
    h, w = frame.shape[:2]
    # 將影像轉為 blob；加大尺寸可提升遠距小物件偵測
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (blob_size, blob_size), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(out_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence >= conf_thresh:
                # 轉回原圖座標
                center_x, center_y, width, height = detection[0:4]
                x = int((center_x - width / 2) * w)
                y = int((center_y - height / 2) * h)
                bw = int(width * w)
                bh = int(height * h)
                boxes.append([x, y, bw, bh])
                confidences.append(confidence)
                class_ids.append(class_id)

    # NMS 去重疊
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
    results = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, bw, bh = boxes[i]
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            cls_name = classes[class_ids[i]] if 0 <= class_ids[i] < len(classes) else str(class_ids[i])
            conf = confidences[i]
            if cls_name in VEHICLE_CLASSES:
                results.append((x1, y1, x2, y2, cls_name, conf))
    return results


def combine_nms(detections, nms_thresh=0.5):
    # detections: list of (x1,y1,x2,y2,cls,conf)
    if not detections:
        return []
    boxes = []
    confidences = []
    for x1, y1, x2, y2, cls, conf in detections:
        boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
        confidences.append(float(conf))
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.0001, nms_threshold=nms_thresh)
    out = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, bw, bh = boxes[i]
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            out.append((x1, y1, x2, y2, detections[i][4], detections[i][5]))
    return out


def main():
    print("Loading YOLOv4-tiny via OpenCV DNN...")
    net, out_layers, classes = load_yolo()

    print(f"Opening stream: {STREAM_URL}")

    prev_time = time.time()
    fps = 0.0

    for frame in mjpeg_frames(STREAM_URL):
        # 第一階偵測：整張影像使用較大 blob（提升遠距偵測能力）
        det_full = detect_vehicles(frame, net, out_layers, classes, conf_thresh=0.22, nms_thresh=0.5, blob_size=832)

        # 第二階偵測：針對遠處 ROI 放大處理
        rx, ry, rw, rh = SECOND_PASS_ROI
        H, W = frame.shape[:2]
        x1 = int(rx * W)
        y1 = int(ry * H)
        x2 = int((rx + rw) * W)
        y2 = int((ry + rh) * H)
        roi = frame[y1:y2, x1:x2]
        det_roi = []
        if roi.size > 0:
            det_r = detect_vehicles(roi, net, out_layers, classes, conf_thresh=0.20, nms_thresh=0.5, blob_size=768)
            # 映射回原座標
            for xr1, yr1, xr2, yr2, cls_name, conf in det_r:
                det_roi.append((xr1 + x1, yr1 + y1, xr2 + x1, yr2 + y1, cls_name, conf))

        # 合併並做一次 NMS 去重疊
        detections = combine_nms(det_full + det_roi, nms_thresh=0.5)

        # FPS
        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # 畫框
        for x1, y1, x2, y2, cls_name, conf in detections:
            label = f"{cls_name} {conf:.2f}"
            draw_label(frame, x1, y1, x2, y2, label, (0, 255, 0))

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # 顯示前先放大
        if DISPLAY_SCALE and DISPLAY_SCALE > 1:
            disp = cv2.resize(frame, (frame.shape[1] * DISPLAY_SCALE, frame.shape[0] * DISPLAY_SCALE), interpolation=cv2.INTER_LINEAR)
        else:
            disp = frame

        # 可調整視窗大小
        win_name = "Freeway Vehicle Detection (Press q to quit)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            globals()['DISPLAY_SCALE'] = 1
        elif key == ord('2'):
            globals()['DISPLAY_SCALE'] = 2
        elif key == ord('3'):
            globals()['DISPLAY_SCALE'] = 3
        elif ZOOM_KEYS and key == ord('+'):
            globals()['DISPLAY_SCALE'] = min(4, globals()['DISPLAY_SCALE'] + 1)
        elif ZOOM_KEYS and key == ord('-'):
            globals()['DISPLAY_SCALE'] = max(1, globals()['DISPLAY_SCALE'] - 1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. Exiting...")
        sys.exit(0)
