# YOLO Object Detection - Keerthi
# Install first if needed: pip install ultralytics opencv-python

from ultralytics import YOLO
import cv2
def detect_image(image_path):
    model = YOLO("yolov8n.pt")  # pre-trained model
    results = model(image_path)
    results[0].show()
    results.save("output/")
    print("Output saved in 'output/' folder.")
def detect_webcam():
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated = results[0].plot()
        cv2.imshow("YOLOv8 Detection - Keerthi", annotated)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    # Example: detect_image("input.jpg")
    print("YOLO Object Detection Script - Keerthi")
