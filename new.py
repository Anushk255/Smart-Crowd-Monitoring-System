import cv2
import torch

# YOLOv5 model load karo (pre-trained)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Camera start karo (0 = webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detection
    results = model(frame)

    # Results dataframe
    df = results.pandas().xyxy[0]

    person_count = 0

    for index, row in df.iterrows():
        if row['name'] == 'person':  # sirf log count karna
            person_count += 1

            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

            # Bounding box draw karo
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    # Count display
    cv2.putText(frame, f'People Count: {person_count}', (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Crowd Monitoring System", frame)

    # Exit key (q)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()