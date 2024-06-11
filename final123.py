from ultralytics import YOLO
import cv2
import cvzone
import math
import pyttsx3
text_speech = pyttsx3.init()
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
model = YOLO("../yolow/yolov8n.pt")
classNames = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy", "bear", "hair drier", "toothbrush"]
while True:
    success, img = cap.read()
    results = model(img, stream=True)
    detected_objects = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            # confidence
            conf = math.ceil(box.conf[0] * 100) / 100
            # class name
            cls = int(box.cls[0])
            class_name = classNames[cls]
            detected_objects.append((class_name, conf, (x1, y1, x2, y2)))
            cvzone.putTextRect(img, f'{class_name} {conf}', (max(0, x1), max(35, y1)))
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    # Process detected objects and provide navigation instructions
    navigation_instructions = []
    for obj, conf, box in detected_objects:
        if obj == "person":
            if box[0] < img.shape[1] / 3:
                navigation_instructions.append("Turn left.")
            elif box[2] > 2 * img.shape[1] / 3:
                navigation_instructions.append("Turn right.")
            else:
                navigation_instructions.append("Move forward.")
    # Display the navigation instructions on the console 
    if navigation_instructions:
        text_speech.say(navigation_instructions)
        text_speech.runAndWait()
        instruction_text = " ".join(navigation_instructions)
        print(instruction_text)
    if key == 49:
        cv2.destroyAllWindows()
        break