import torch
import cv2
import numpy as np

# Load the YOLOv5 model
model_path = 'yolov5s.pt'  # Replace with the actual path to your model file
model = torch.load(model_path, map_location=torch.device('cpu'))['model'].float()  # Cast the model to FP32
model.eval()  # Set the model to evaluation mode

# Open a connection to the camera (0 for default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to a PyTorch tensor
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    # Run the frame through the YOLOv5 model
    with torch.no_grad():  # Disable gradient calculation
        results = model(img_tensor)

    # Access the predictions
    predictions = results[0].cpu().numpy()  # Get the detection results and move to CPU

    # Process each detected obstacle
    for box in predictions[0]:  # Iterate over detections
        # Extract bounding box and confidence
        x1, y1, x2, y2, confidence = box[:5]
        
        # Convert to scalar values
        x1, y1, x2, y2, confidence = float(x1), float(y1), float(x2), float(y2), float(confidence)

        # Check if confidence is above a threshold (e.g., 0.5)
        if confidence > 0.5:
            # Calculate the center of the bounding box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Determine the direction of the obstacle
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

            # Print an alert with guidance based on obstacle location
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

            # Draw the bounding box and direction on the original frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, direction, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Camera Feed with Obstacles', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
