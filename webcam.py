from PIL import Image
import cv2
import torch
import math 
import function.utils_rotate as utils_rotate
from IPython.display import display
import os
import time
import argparse
import function.helper as helper

# Load model
yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector_nano_61.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr_nano_62.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60

prev_frame_time = 0
new_frame_time = 0

# Tạo thư mục lưu
os.makedirs("output/images", exist_ok=True)
log_file_path = "output/log.txt"

vid = cv2.VideoCapture(0)
frame_count = 0

while True:
    ret, frame = vid.read()
    if not ret or frame is None:
        print("❌ Không đọc được frame từ webcam. Thử kiểm tra lại camera index hoặc thiết bị.")
        break

    plates = yolo_LP_detect(frame, size=640)
    list_plates = plates.pandas().xyxy[0].values.tolist()
    list_read_plates = set()

    for plate in list_plates:
        flag = 0
        x = int(plate[0])
        y = int(plate[1])
        w = int(plate[2] - plate[0])
        h = int(plate[3] - plate[1])
        crop_img = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (int(plate[0]), int(plate[1])), (int(plate[2]), int(plate[3])), color=(0, 0, 225), thickness=2)
        cv2.imwrite("crop.jpg", crop_img)
        rc_image = cv2.imread("crop.jpg")
        lp = ""
        for cc in range(0, 2):
            for ct in range(0, 2):
                lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                if lp != "unknown":
                    list_read_plates.add(lp)
                    cv2.putText(frame, lp, (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    flag = 1
                    break
            if flag == 1:
                break

    # Hiển thị FPS và hướng dẫn
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time + 1e-8)
    prev_frame_time = new_frame_time
    fps = int(fps)
    cv2.putText(frame, f"FPS: {fps}", (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, "Nhan 's' de luu, 'q' de thoat", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF

    # Nhấn 's' để lưu ảnh + ghi vào 1 file log duy nhất
    if key == ord('s') and list_read_plates:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename_img = f"output/images/frame_{timestamp}.jpg"

        # Lưu ảnh
        cv2.imwrite(filename_img, frame)

        # Ghi log vào file duy nhất
        with open(log_file_path, "a", encoding="utf-8") as f_log:
            f_log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")
            f_log.write(f"Ảnh: {filename_img}\n")
            f_log.write("Biển số nhận dạng được:\n")
            for lp in list_read_plates:
                f_log.write(f"- {lp}\n")
            f_log.write("\n")

        print(f"\n[INFO] Đã lưu:\n- Ảnh: {filename_img}\n- Ghi vào: {log_file_path}")

    # Nhấn 'q' để thoát
    if key == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()





# Học có giám sát hay không giám sát