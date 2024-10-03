import cv2
import time
from ultralytics import YOLO

# Muat model ONNX
model = YOLO("yolo11n.onnx")  # Pastikan path ini benar

# Inisialisasi webcam (biasanya device 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Tidak dapat membuka webcam.")
    exit()

# Inisialisasi variabel untuk menghitung FPS
prev_time = 0
fps = 0

# Loop untuk membaca frame dari webcam
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Tidak dapat membaca frame dari webcam.")
        break

    # Hitung FPS
    current_time = time.time()
    elapsed_time = current_time - prev_time
    prev_time = current_time
    if elapsed_time > 0:
        fps = 1 / elapsed_time

    # Jalankan inferensi pada frame
    results = model(frame)

    # Render hasil deteksi pada frame
    annotated_frame = results[0].plot()

    # Tambahkan teks FPS pada frame
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (10, 30),  # Posisi teks (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,  # Font
        1,  # Skala font
        (0, 255, 0),  # Warna (B, G, R)
        2,  # Ketebalan garis
        cv2.LINE_AA  # Jenis garis
    )

    # Tampilkan frame dengan deteksi dan FPS
    cv2.imshow("Webcam Real-time YOLO Detection", annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan resource
cap.release()
cv2.destroyAllWindows()
