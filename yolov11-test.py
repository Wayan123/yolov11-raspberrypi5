import cv2
import time
from ultralytics import YOLO
import torch

def realtime_detection(imgsz=640):
    """
    Fungsi untuk menjalankan deteksi objek secara real-time dengan YOLOv11,
    menggunakan ukuran gambar input imgsz dan menampilkan FPS.
    
    Parameters:
    - imgsz (int): Ukuran dimensi input gambar (harus kelipatan dari 32).
                  Contoh: 640 atau 1280
    """
    # Cek apakah GPU tersedia
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device yang digunakan untuk inferensi: {device}')

    # Inisialisasi model YOLOv11
    try:
        model = YOLO("yolo11l-seg.pt")  # Path ke model yang telah dilatih
        model.to(device)  # Memindahkan model ke GPU jika tersedia
        print("Model YOLOv11 berhasil dimuat untuk inferensi.")
    except Exception as e:
        print(f"Gagal memuat model YOLOv11: {e}")
        return

    # Memastikan ukuran gambar adalah kelipatan dari 32
    if imgsz % 32 != 0:
        print("Ukuran gambar harus merupakan kelipatan dari 32. Menyesuaikan ke kelipatan 32 terdekat.")
        imgsz = (imgsz // 32) * 32
        print(f'Ukuran gambar input disesuaikan ke: {imgsz}x{imgsz}')

    print(f'Menggunakan ukuran gambar input: {imgsz}x{imgsz}')
    # Beberapa model YOLO mungkin membutuhkan resize manual, tetapi ultralytics YOLO biasanya mengatur ini secara internal

    # Inisialisasi webcam (gunakan 0 untuk kamera default)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Tidak dapat membuka kamera")
        return

    # Mengatur resolusi kamera untuk menghemat bandwidth dan mempercepat proses
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, imgsz)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, imgsz)

    # Variabel untuk menghitung FPS
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera")
            break

        # Memulai timer untuk FPS
        start_time = time.time()

        # Melakukan inferensi pada frame dengan imgsz tertentu
        try:
            results = model(frame, conf=0.5, imgsz=imgsz, verbose=False)[0]  # Mengatur confidence threshold ke 50%
        except Exception as e:
            print(f"Error saat menjalankan inferensi: {e}")
            break

        # Menggambar hasil deteksi pada frame
        annotated_frame = results.plot()

        # Menghitung FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)

        # Menampilkan FPS pada frame
        cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Menampilkan frame dengan deteksi
        cv2.imshow('YOLOv11 Real-time Detection', annotated_frame)

        # Keluar dari loop jika tombol 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Keluar dari aplikasi.")
            break

    # Melepaskan sumber daya
    cap.release()
    cv2.destroyAllWindows()

def main():
    """
    Fungsi utama untuk menjalankan deteksi objek real-time dengan YOLOv11.
    Meminta pengguna untuk memilih ukuran gambar input.
    """
    print("Menjalankan Deteksi Objek Real-time dengan YOLOv11")
    print("Pilih ukuran gambar input:")
    print("1. 640x640")
    print("2. 1280x1280")
    print("3. Kustom")

    pilihan = input("Masukkan pilihan (1/2/3): ")

    if pilihan == '1':
        imgsz = 640
    elif pilihan == '2':
        imgsz = 1280
    elif pilihan == '3':
        try:
            imgsz = int(input("Masukkan ukuran gambar input (harus kelipatan dari 32, misalnya 640 atau 1280): "))
            if imgsz % 32 != 0:
                print("Ukuran gambar harus merupakan kelipatan dari 32. Menyesuaikan ke kelipatan 32 terdekat.")
                imgsz = (imgsz // 32) * 32
                print(f'Ukuran gambar input disesuaikan ke: {imgsz}x{imgsz}')
        except ValueError:
            print("Input tidak valid. Menggunakan ukuran gambar default 640x640.")
            imgsz = 640
    else:
        print("Pilihan tidak valid. Menggunakan ukuran gambar default 640x640.")
        imgsz = 640

    realtime_detection(imgsz=imgsz)

if __name__ == "__main__":
    main()
