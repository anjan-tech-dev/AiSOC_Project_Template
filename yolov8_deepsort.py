import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))

import cv2
import easyocr
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize OCR and YOLO
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
model = YOLO('yolov8n.pt')  # Uses pretrained YOLOv8 nano
tracker = DeepSort(max_age=30)

# Define 2-wheeler and 4-wheeler classes based on YOLO's naming
TWO_WHEELERS = ['motorbike', 'motorcycle']
FOUR_WHEELERS = ['car', 'truck', 'bus']


# # 🔧 Low-light enhancement function
def enhance_frame(frame):
    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # Split into L, A, and B channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L-channel (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge enhanced L with original A and B
    enhanced_lab = cv2.merge((cl, a, b))

    # Convert back to BGR color space
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Optional: Noise reduction
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)

    return enhanced

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0,
                          (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # enhanced_frame = enhance_frame(frame) 
        results = model(frame)[0]
        detections = []

        # Get vehicle detections from YOLO
        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = model.model.names[cls_id].lower()

            if class_name not in TWO_WHEELERS + FOUR_WHEELERS:
                continue  # Skip everything else (like persons, animals, etc.)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if conf < 0.4:  # or your tuned threshold
                continue

            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, class_name))

        # Track using DeepSORT
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            # Get bounding box
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            w, h = x2 - x1, y2 - y1
            class_name = track.get_det_class().lower()

            # Crop and OCR for plate
            cropped = frame[y1:y2, x1:x2]
            plate = ''
            try:
                ocr_result = reader.readtext(cropped)
                if ocr_result:
                    plate = ocr_result[0][1]
            except:
                pass

            # Set label and color
            label = f"{class_name.upper()} | {plate}" if plate else class_name.upper()
            color = (0, 0, 255) if class_name in TWO_WHEELERS else (0, 255, 0)

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # cv2.putText(frame, label, (x1, y1 - 15),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            # Define label position and size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = x1
            text_y = y1 + text_size[1] + 5  # Inside the box, near top

            # Draw background rectangle for text
            cv2.rectangle(frame,
                        (text_x, text_y - text_size[1] - 5),
                        (text_x + text_size[0] + 4, text_y + 4),
                        color,  # Same as box color
                        thickness=cv2.FILLED)

            # Put the label text on top of the rectangle
            cv2.putText(frame, label, (text_x + 2, text_y),
                        font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)


        out.write(frame)

    cap.release()
    out.release()
