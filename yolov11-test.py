import cv2
import time
from ultralytics import YOLO
import torch

def realtime_detection(imgsz=640):
    """
    Function to perform real-time object detection with YOLOv11,
    using the input image size imgsz and displaying FPS.
    
    Parameters:
    - imgsz (int): Input image dimension size (must be a multiple of 32).
                  Example: 640 or 1280
    """
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device used for inference: {device}')

    # Initialize YOLOv11 model
    try:
        model = YOLO("yolo11l-seg.pt")  # Path to the trained model
        model.to(device)  # Move the model to GPU if available
        print("YOLOv11 model successfully loaded for inference.")
    except Exception as e:
        print(f"Failed to load YOLOv11 model: {e}")
        return

    # Ensure image size is a multiple of 32
    if imgsz % 32 != 0:
        print("Image size must be a multiple of 32. Adjusting to the nearest multiple of 32.")
        imgsz = (imgsz // 32) * 32
        print(f'Input image size adjusted to: {imgsz}x{imgsz}')

    print(f'Using input image size: {imgsz}x{imgsz}')
    # Some YOLO models may require manual resizing, but ultralytics YOLO usually handles this internally

    # Initialize webcam (use 0 for default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Set camera resolution to save bandwidth and speed up processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, imgsz)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, imgsz)

    # Variable for calculating FPS
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera")
            break

        # Start timer for FPS
        start_time = time.time()

        # Perform inference on frame with a certain imgsz
        try:
            results = model(frame, conf=0.5, imgsz=imgsz, verbose=False)[0]  # Set confidence threshold to 50%
        except Exception as e:
            print(f"Error during inference: {e}")
            break

        # Draw detection results on frame
        annotated_frame = results.plot()

        # Calculate FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)

        # Display FPS on frame
        cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display frame with detections
        cv2.imshow('YOLOv11 Real-time Detection', annotated_frame)

        # Exit loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting application.")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def main():
    """
    Main function to run real-time object detection with YOLOv11.
    Prompts the user to choose the input image size.
    """
    print("Running Real-time Object Detection with YOLOv11")
    print("Choose input image size:")
    print("1. 640x640")
    print("2. 1280x1280")
    print("3. Custom")

    choice = input("Enter your choice (1/2/3): ")

    if choice == '1':
        imgsz = 640
    elif choice == '2':
        imgsz = 1280
    elif choice == '3':
        try:
            imgsz = int(input("Enter input image size (must be a multiple of 32, e.g., 640 or 1280): "))
            if imgsz % 32 != 0:
                print("Image size must be a multiple of 32. Adjusting to the nearest multiple of 32.")
                imgsz = (imgsz // 32) * 32
                print(f'Input image size adjusted to: {imgsz}x{imgsz}')
        except ValueError:
            print("Invalid input. Using default image size 640x640.")
            imgsz = 640
    else:
        print("Invalid choice. Using default image size 640x640.")
        imgsz = 640

    realtime_detection(imgsz=imgsz)

if __name__ == "__main__":
    main()
