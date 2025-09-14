# Helmet Detection using YOLOv8

This project detects whether a person is wearing a helmet or not using YOLOv8 and OpenCV.  
It can be used for traffic surveillance, rider safety monitoring, and smart city applications.

## 🚀 Features Dataset
- Detects helmets in images and videos
- YOLOv8 trained on custom dataset
- Real-time detection using webcam or video file
- Outputs annotated video with bounding boxes

## 🛠 Tech Stack
- Python 3
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy

## ⚙ Installation
Clone the repository and install dependencies:
```
bash
git clone ## ⚙ Installation
Clone the repository and install dependencies:

bash
git clone https://github.com/your-username/helmet-detection.git
cd helmet-detection
pip install -r requirements.txt
```
## 📂 Dataset 
The dataset used for training contains images of riders with and without helmets.  
Label files are stored in `train/labels/`.  

You can prepare your dataset using **LabelImg** or **Roboflow**.

## ▶ Usage

### Run detection on an image:
```
bash
python main.py --source input_video.mp4

```
## 📊 Results
Example detection:
(Input image → Output with bounding boxes and labels)
![image alt](https://github.com/bhavana239/helmet-detection/blob/8224d6c6cce58ee5037cab0a03cd00efdace746e/hd1.png)
![image alt](https://github.com/bhavana239/helmet-detection/blob/dcd3995c8bcb39ea3e72d7d653301bdd0ec0aa5c/hd2.png)
