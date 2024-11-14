import torch
import cv2
import numpy as np

model_path = 'yolov5s.pt'
model = torch.load(model_path, map_location=torch.device('cpu'))['model'].float()
model.eval()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    with torch.no_grad():
        results = model(img_tensor)

    predictions = results[0].cpu().numpy()

    for box in predictions[0]:
        x1, y1, x2, y2, confidence = box[:5]
        x1, y1, x2, y2, confidence = float(x1), float(y1), float(x2), float(y2), float(confidence)

        if confidence > 0.5:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            direction = ""
            if center_x < frame.shape[1] / 3:
                direction = "Left"
            elif center_x > 2 * frame.shape[1] / 3:
                direction = "Right"
            else:
                direction = "Center"

            if center_y < frame.shape[0] / 3:
                direction += " Top"
            elif center_y > 2 * frame.shape[0] / 3:
                direction += " Bottom"
            else:
                direction += " Middle"

            if "Left" in direction:
                print("Alert: Obstacle on the LEFT. Move RIGHT to avoid.")
            elif "Right" in direction:
                print("Alert: Obstacle on the RIGHT. Move LEFT to avoid.")
            elif "Center" in direction:
                if "Top" in direction:
                    print("Alert: Obstacle directly AHEAD at the TOP. Caution while moving forward.")
                elif "Bottom" in direction:
                    print("Alert: Obstacle in front at the BOTTOM. Be cautious.")
                else:
                    print("Alert: Obstacle directly AHEAD. Consider slowing down or adjusting position.")

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, direction, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('Camera Feed with Obstacles', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
