# Nhận diện và quản lý biển số xe Việt Nam bằng YOLOv5

> Đồ án môn **Công nghệ đa phương tiện** – Khoa Công nghệ Thông tin, Trường Đại học Mỏ – Địa chất (5/2025)
> Đề tài: *Ứng dụng AI trong việc nhận diện và quản lý biển số xe trong các nơi gửi xe với quy mô lớn*
> Sinh viên thực hiện: **Đinh Mai An** · Giảng viên hướng dẫn: **ThS. Dương Chí Thiện**

> 📌 **Nguồn gốc mã nguồn:** repo này là bản fork của [trungdinh22/License-Plate-Recognition](https://github.com/trungdinh22/License-Plate-Recognition). Toàn bộ mô hình đã huấn luyện, notebook huấn luyện, các hàm xử lý (`function/`) và notebook minh họa `LP_recognition.ipynb` thuộc về tác giả gốc. Phần do mình bổ sung được liệt kê ở mục [Phần bổ sung trong đồ án](#phần-bổ-sung-trong-đồ-án).

---

## Giới thiệu

Hệ thống phát hiện và đọc biển số xe Việt Nam (biển 1 dòng và 2 dòng) từ ảnh tĩnh hoặc webcam theo thời gian thực, hướng tới ứng dụng ghi nhận xe ra vào tại các bãi gửi xe.

Quy trình xử lý gồm 3 bước:

1. **Phát hiện biển số** – mô hình YOLOv5 (`LP_detector`) khoanh vùng biển số trong khung hình.
2. **Làm thẳng ảnh (deskew)** – xoay, căn chỉnh vùng biển số bị nghiêng trước khi đọc.
3. **Nhận dạng ký tự** – mô hình YOLOv5 thứ hai (`LP_ocr`) phát hiện từng ký tự, sau đó sắp xếp thành chuỗi biển số (xử lý cả biển 2 dòng).

![Kết quả](result/image.jpg)

## Phần bổ sung trong đồ án

So với mã nguồn gốc, mình đã chỉnh sửa và bổ sung để phục vụ bài toán **quản lý xe tại bãi gửi**:

- **Lưu kết quả nhận diện:** ảnh kết quả được lưu vào `output/images/` kèm dấu thời gian.
- **Ghi nhật ký biển số:** mỗi lần nhận diện được ghi vào `output/log.txt` gồm thời gian, đường dẫn ảnh và danh sách biển số đọc được. Đây là cơ sở để tra cứu xe ra/vào.
- **Điều khiển khi chạy webcam:** nhấn `s` để lưu khung hình và ghi log, nhấn `q` để thoát; hiển thị hướng dẫn và FPS trên màn hình.
- **Xử lý lỗi:** dừng an toàn và báo lỗi khi không đọc được khung hình từ webcam; sửa lỗi chia cho 0 khi tính FPS.
- **Tinh gọn `lp_image.py`:** bỏ bước ghi/đọc lại file ảnh tạm khi cắt biển số, thêm chú thích tiếng Việt.
- Viết **báo cáo** và **slide thuyết trình** (thư mục [`docs/`](docs/)).

Ví dụ nội dung `output/log.txt`:

```
[2025-05-06 16:09:00]
Ảnh: output/images/frame_20250506_160900.jpg
Biển số nhận dạng được:
- 59G2-44455
```

## Công nghệ sử dụng

- **Python**, **PyTorch**, **YOLOv5**: phát hiện biển số và ký tự
- **OpenCV**: đọc ảnh/webcam, xử lý ảnh, vẽ kết quả
- **NumPy, Pandas, Pillow**

## Cài đặt

```bash
git clone https://github.com/maian2309/License-Plate-Recognition.git
cd License-Plate-Recognition
pip install -r ./requirement.txt
```

- Mô hình đã huấn luyện có sẵn trong thư mục `./model` (của tác giả gốc).
- Tải YOLOv5 (bản cũ) tại: [yolov5 – Google Drive](https://drive.google.com/file/d/1g1u7M4NmWDsMGOppHocgBKjbwtDA-uIu/view?usp=sharing), rồi chép thư mục `yolov5` vào thư mục dự án.

## Chạy chương trình

```bash
# Nhận diện từ webcam (nhấn 's' để lưu, 'q' để thoát)
python webcam.py

# Nhận diện từ ảnh tĩnh
python lp_image.py -i test_image/3.jpg
```

Kết quả được lưu trong thư mục `output/` (đã được loại khỏi Git bằng `.gitignore`).

![Demo](result/video_1.gif)

## Dữ liệu và huấn luyện

Theo tác giả gốc, hệ thống dùng 2 bộ dữ liệu cho 2 giai đoạn:

- [License Plate Detection Dataset](https://drive.google.com/file/d/1xchPXf7a1r466ngow_W_9bittRqQEf_T/view?usp=sharing)
- [Character Detection Dataset](https://drive.google.com/file/d/1bPux9J0e1mz-_Jssx4XX1-wPGamaS8mI/view?usp=sharing)

Notebook huấn luyện YOLOv5: `training/Plate_detection.ipynb` (phát hiện biển số) và `training/Letter_detection.ipynb` (phát hiện ký tự).

## Cấu trúc thư mục

```
├── function/            # Hàm đọc biển số, làm thẳng ảnh (tác giả gốc)
├── model/               # Mô hình YOLOv5 đã huấn luyện (tác giả gốc)
├── training/            # Notebook huấn luyện (tác giả gốc)
├── test_image/          # Ảnh thử nghiệm
├── result/              # Ảnh/GIF minh họa kết quả
├── docs/                # Báo cáo và slide đồ án
├── lp_image.py          # Nhận diện từ ảnh + lưu kết quả, ghi log
├── webcam.py            # Nhận diện thời gian thực + lưu kết quả, ghi log
└── LP_recognition.ipynb # Minh họa từng bước xử lý (tác giả gốc)
```

## Hướng phát triển

- Lưu lịch sử xe ra/vào vào cơ sở dữ liệu (SQLite/PostgreSQL) thay cho file log, tính thời gian gửi xe và phí.
- Xây dựng giao diện quản lý (web hoặc desktop) để tra cứu biển số theo thời gian.
- Nâng cấp lên YOLOv8 hoặc các phiên bản mới, tối ưu cho điều kiện thiếu sáng và biển số bị che khuất.

## Ghi công

- Mã nguồn, mô hình và hướng dẫn huấn luyện gốc: [trungdinh22/License-Plate-Recognition](https://github.com/trungdinh22/License-Plate-Recognition)
- Một phần dữ liệu do [Mì Ai](https://www.miai.vn/thu-vien-mi-ai/) và [winter2897](https://github.com/winter2897/Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano/blob/main/doc/dataset.md) chia sẻ (theo tác giả gốc).
