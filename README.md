
# 💤 Real-Time Drowsiness & Blink Detection

This project implements a **real-time eye monitoring system** to detect **blinks** and **drowsiness** using the webcam feed. By leveraging **Eye Aspect Ratio (EAR)** with **OpenCV**, **Dlib**, and **Numpy**, the system alerts when the user's eyes are closed for too long — a common sign of drowsiness.

---

## 🚀 Features

- 👁️ Real-time **blink detection**
- 🛌 Detects **drowsiness** based on sustained low EAR values
- 🧠 Uses **facial landmarks** for precise eye tracking
- 📷 **Live webcam** feedback with drawn eye contours
- ⚠️ Customizable EAR thresholds and drowsy frame counts
- 🔴 Visual alerts displayed directly on video feed

---

## 🧠 How It Works

1. **🎥 Video Capture**  
   Accesses the webcam and reads live frames.

2. **🧍 Face Detection**  
   Detects faces using **Dlib's HOG-based detector**.

3. **👀 Eye Aspect Ratio (EAR) Calculation**  
   Computes EAR using the 6 key eye landmarks for both eyes.

4. **💤 Drowsiness & Blink Detection**  
   - EAR drops below a certain threshold ➝ considered a **blink**  
   - EAR remains low across several frames ➝ **drowsy alert** is triggered

5. **📺 Real-time Display & Alert**  
   - Draws contours around eyes  
   - Displays status messages like "You are blinking." or "You are drowsy!"
   - If the user's eyes remain closed for 10 seconds or more, the system triggers a looping alert  
     sound to warn the user.

6. **❌ Exit Option**  
   Press `q` to gracefully exit the application.

---

## 🗂️ Folder Structure

```
Drowsiness Detection System/
    ├── Alert/
    |   └──alert.wav
    ├── Dataset/
    │   └── shape_predictor_68_face_landmarks.dat   # Pre-trained model for facial landmarks
    └── main.py                                     # Main detection script
```

---

## ⚙️ Installation

### 🐍 Python Dependencies:

Make sure you have Python installed, then install the required libraries:

```bash
pip install opencv-python dlib imutils numpy
```

## ▶️ Run the Project

### 📥 Clone the Repository:

```bash
git clone https://github.com/pbhalange/Drowsiness-Detection-System.git
cd Drowsiness-Detection-System\
```

---

```bash
python main.py
```

Make sure the `shape_predictor_68_face_landmarks.dat` file is in the `Dataset/` directory.

---

## 🛠️ Customization

You can tweak the detection sensitivity by changing these values inside `main.py`:

| Parameter             | Description                                              | Default Value |
|-----------------------|----------------------------------------------------------|---------------|
| `MIN_EAR`             | Minimum EAR to consider as a **blink**                  | `0.2`         |
| `MIN_DROWSY_EAR`      | EAR threshold for **drowsiness detection**              | `0.3`         |
| `MAX_DROWSY_FRAMES`   | Number of frames of low EAR before triggering alert     | `35`          |

---


## Credits

- Dlib's [68-point facial landmark model](http://dlib.net/files/)
- OpenCV for image processing
---

