import cv2
import os
import csv
from ultralytics import YOLO
import numpy as np # Import numpy for potential array operations if needed

# --- Configuration Section ---
# Path to your trained YOLO model weights
MODEL_PATH = r"C:\Users\samee\runs\detect\train7\weights\best.pt"

# Path to your input video file
VIDEO_PATH = "detector.mp4"

# Directory to save violation images
VIOLATIONS_DIR = "violations"

# CSV log file name
LOG_FILE_NAME = "violation_log.csv"

# Minimum confidence threshold for a detection to be considered valid and drawn
# Start with a lower value (e.g., 0.1 or 0.05) for debugging to see all detections.
# Increase later to reduce false positives.
CONFIDENCE_THRESHOLD = 0.01 # A common production threshold, but try 0.05 for debugging

# --- Main Code ---

try:
    # Load trained YOLO model
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully.")
    print("✅ MODEL CLASSES:", model.names)
except Exception as e:
    print(f"❌ Error loading YOLO model from {MODEL_PATH}: {e}")
    print("Please ensure the model path is correct and the file exists.")
    exit()

# Create directory to save violation images
try:
    os.makedirs(VIOLATIONS_DIR, exist_ok=True)
    print(f"✅ Violation directory '{VIOLATIONS_DIR}' ensured.")
except Exception as e:
    print(f"❌ Error creating violations directory '{VIOLATIONS_DIR}': {e}")
    exit()

# Load input video
cap = cv2.VideoCapture(VIDEO_PATH)

# Check if video opened successfully
if not cap.isOpened():
    print(f"❌ Error: Could not open video file '{VIDEO_PATH}'.")
    print("Please check the video path and ensure the file exists and is not corrupted.")
    print("You might also need to install appropriate video codecs if not already present.")
    exit()
else:
    print(f"✅ Video '{VIDEO_PATH}' opened successfully.")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video Resolution: {frame_width}x{frame_height}, FPS: {fps:.2f}")


# Prepare CSV file for logging violations
try:
    log_file = open(LOG_FILE_NAME, mode="w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["Frame", "Time (s)", "Label", "Confidence", "Bounding Box (x1,y1,x2,y2)"])
    print(f"✅ Log file '{LOG_FILE_NAME}' opened for writing.")
except Exception as e:
    print(f"❌ Error opening log file '{LOG_FILE_NAME}': {e}")
    cap.release()
    exit()

frame_number = 0

print("\nStarting video processing. Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("End of video stream or failed to read frame.")
        break

    frame_number += 1
    time_stamp = frame_number / fps
    
    # It's important to work on a copy of the frame if you modify it,
    # to avoid unexpected side effects if 'frame' is used elsewhere.
    # For simple display, directly modifying 'frame' is also fine,
    # but using 'annotated_frame' is clearer.
    annotated_frame = frame.copy() 

    # Predict using YOLO
    # Setting verbose=True here might give you more console output from YOLO itself
    results = model.predict(source=annotated_frame, conf=CONFIDENCE_THRESHOLD, imgsz=640, verbose=False)

    # Process each result (usually one per image/frame)
    for result in results:
        # Iterate over each detected bounding box
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            cls_id = int(box.cls[0])                # Class ID
            confidence = float(box.conf[0])         # Confidence score
            
            # Ensure the class ID exists in model.names to prevent KeyError
            if cls_id not in model.names:
                print(f"⚠ Warning: Unknown class ID {cls_id} detected. Skipping drawing for this box.")
                continue
            
            label = model.names[cls_id].lower()     # Get class label string (e.g., "helmet", "no_helmet")

            # --- Debugging Print (CRITICAL) ---
            # This line is crucial for verifying if detections are happening.
            print(f"[Frame {frame_number:05d}] Detected: {label} (Conf: {confidence:.2f}) at [{x1},{y1},{x2},{y2}]")
            # --- End Debugging Print ---

            # Initialize drawing parameters
            color = (0, 0, 0)  # Default color (black)
            display_text = "Unknown" # Text to display on the frame
            text_color = (255, 255, 255) # Default text color (white)

            # Determine color and text based on the detected label
            if label == "no_helmet":
                color = (0, 0, 255)  # Red for 'No Helmet' violation
                display_text = f"No Helmet ({confidence:.2f})"
                text_color = (255, 255, 255) # White text for better visibility on red background
                
                # Save violation frame and log to CSV
                filename = os.path.join(VIOLATIONS_DIR, f"frame_{frame_number:05d}.jpg")
                try:
                    cv2.imwrite(filename, annotated_frame)
                    csv_writer.writerow([frame_number, round(time_stamp, 2), "No Helmet", 
                                         f"{confidence:.2f}", f"({x1},{y1},{x2},{y2})"])
                except Exception as e:
                    print(f"❌ Error saving image {filename} or writing to CSV: {e}")

            elif label == "helmet":
                color = (0, 255, 0)  # Green for 'Helmet' (safe)
                display_text = f"With Helmet ({confidence:.2f})"
                text_color = (0, 0, 0) # Black text for better visibility on green background
                
                # Log 'With Helmet' to CSV (optional, but good for full record)
                csv_writer.writerow([frame_number, round(time_stamp, 2), "With Helmet", 
                                     f"{confidence:.2f}", f"({x1},{y1},{x2},{y2})"])

            else:
                # If you want to draw other classes with a default style, you can add
                # an 'else' block here. For now, we continue to skip unhandled labels.
                continue 

            # --- Drawing Logic ---
            # Draw the bounding box rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Define font parameters for the text label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2

            # Calculate the size of the text to create a background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(display_text, font, font_scale, font_thickness)

            # Determine the text's Y-position, ensuring it's not drawn off-screen at the top
            # We add a small padding (5 pixels) to the calculated text height
            text_y_position = max(y1 - 10, text_height + 5) 
            
            # Draw a filled rectangle as a background for the text
            # This helps the text stand out regardless of the video content behind it
            cv2.rectangle(annotated_frame, (x1, text_y_position - text_height - baseline), 
                          (x1 + text_width, text_y_position + baseline), color, cv2.FILLED)

            # Put the text label on the frame
            # cv2.LINE_AA adds anti-aliasing for smoother text
            cv2.putText(annotated_frame, display_text, (x1, text_y_position),
                        font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    # Show the modified frame in a window
    cv2.imshow("Helmet Detection", annotated_frame)
    
    # Wait for 1 millisecond. If 'q' is pressed, break the loop.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n'q' pressed. Exiting video stream.")
        break

# --- Cleanup ---
cap.release() # Release video capture object
log_file.close() # Close log file
cv2.destroyAllWindows() # Destroy all OpenCV windows
print("\nProcessing complete. Video stream released, log file closed, and all OpenCV windows destroyed.")