# yolov11-raspberrypi5
Easy to run yolov11 on raspberry pi 5

1. sudo apt update
2. sudo apt upgrade -y
3. sudo apt-get -y install python3-pip libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev
4. python -m venv yolov5
5. source yolov11/bin/activate
6. pip install torch torchvision torchaudio
7. pip install ultralytics
8. pip install opencv-python
9. pip install numpy --upgrade

Test pytorch:
python -c "import torch; print(torch.__version__)"

Then Run Yolov11:
python yolov11-test.py
