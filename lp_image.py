from PIL import Image
import cv2
import torch
import math 
import function.utils_rotate as utils_rotate
import os
import time
import argparse
import function.helper as helper

# Đọc tham số đầu vào
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
args = ap.parse_args()

# Load mô hình
yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60

# Đọc ảnh
img = cv2.imread(args.image)
plates = yolo_LP_detect(img, size=640)
list_plates = plates.pandas().xyxy[0].values.tolist()
list_read_plates = set()

# Nếu không detect được khung biển, thử OCR trực tiếp
if len(list_plates) == 0:
    lp = helper.read_plate(yolo_license_plate, img)
    if lp != "unknown":
        cv2.putText(img, lp, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        list_read_plates.add(lp)
else:
    for plate in list_plates:
        flag = 0
        x = int(plate[0])
        y = int(plate[1])
        w = int(plate[2] - plate[0])
        h = int(plate[3] - plate[1])
        crop_img = img[y:y+h, x:x+w]
        cv2.rectangle(img, (x, y), (x+w, y+h), color=(0, 0, 225), thickness=2)
        for cc in range(0, 2):
            for ct in range(0, 2):
                lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                if lp != "unknown":
                    list_read_plates.add(lp)
                    cv2.putText(img, lp, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    flag = 1
                    break
            if flag == 1:
                break

# Hiển thị ảnh kết quả
cv2.imshow('frame', img)
cv2.waitKey()
cv2.destroyAllWindows()

# === Tạo thư mục lưu kết quả ===
os.makedirs("output/images", exist_ok=True)
log_file_path = "output/log.txt"

# === Lưu ảnh kết quả ===
timestamp = time.strftime('%Y%m%d_%H%M%S')
img_output_path = os.path.join("output/images", f"result_{timestamp}.jpg")
cv2.imwrite(img_output_path, img)
print(f"[INFO] Ảnh kết quả đã lưu tại: {img_output_path}")

# === Ghi vào file log duy nhất ===
with open(log_file_path, "a", encoding="utf-8") as f_log:
    f_log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")
    f_log.write(f"Ảnh: {img_output_path}\n")
    if list_read_plates:
        f_log.write("Biển số nhận dạng được:\n")
        for lp in list_read_plates:
            f_log.write(f"- {lp}\n")
    else:
        f_log.write("Không nhận diện được biển số nào.\n")
    f_log.write("\n")

print(f"[INFO] Ghi log tại: {log_file_path}")
