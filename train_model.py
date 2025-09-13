# # # train_model.py (or whatever you name this file)

# from ultralytics import YOLO

# # Initialize the YOLOv8n model
# # Make sure yolov8n.pt is in the same directory as this script, or provide its full path
# model = YOLO('yolov8n.pt') 

# # Start training the model
# # It will save results in a new folder like 'runs/detect/diagnosis_run_final'
# model.train(
#     data=r'C:\Users\samee\OneDrive\Desktop\helmet_detector\data.yaml', # Path to your data.yaml
#     epochs=100, # Set target epochs to 50
#     imgsz=640, # Input image size
#     name='diagnosis_run_final', # IMPORTANT: This will create a NEW, unique folder for this run's results
#     verbose=True, # IMPORTANT: This gives us detailed output in the log file
#     patience=50 # IMPORTANT: Set patience to match epochs to ensure it doesn't stop early if validation plateaus
# )

# print("Training process initiated. Check the 'runs/detect/diagnosis_run_final' folder for results after completion.")
# train_model.py

from ultralytics import YOLO

model = YOLO('yolov8n.pt') 

model.train(
    data=r'C:\Users\samee\OneDrive\Desktop\helmet_detector\data.yaml',
    epochs=50, # Or 100 if you want to try, but 50 is fine for diagnosis
    imgsz=640,
    name='training_with_augmentation', # NEW folder name for this run
    verbose=True,
    patience=50, # Ensures it runs for full epochs if not overfitting heavily

    # --- Data Augmentation Parameters (adjust these!) ---
    # These are some common ones you can explicitly set or adjust their intensity
    fliplr=0.5,       # Random horizontal flip (0.5 means 50% chance). Common.
    flipud=0.0,       # Random vertical flip (0.0 means no vertical flip). Often keep 0.
    hsv_h=0.015,      # Hue augmentation (range 0-1)
    hsv_s=0.7,        # Saturation augmentation (range 0-1)
    hsv_v=0.4,        # Value (brightness) augmentation (range 0-1)
    degrees=0.0,      # Random rotation (degrees)
    translate=0.1,    # Random translation (fraction of image size)
    scale=0.5,        # Random scaling (min/max scale factor)
    shear=0.0,        # Random shear (degrees)
    perspective=0.0,  # Random perspective distortion (0.0 to 0.001)
    mixup=0.0,        # MixUp augmentation (0.0 to 1.0, typically 0.0 or 0.1)
    copy_paste=0.0,   # Copy-Paste augmentation (0.0 to 1.0, typically 0.0 or 0.1)
    # ---------------------------------------------------
)

print("Training process initiated.")