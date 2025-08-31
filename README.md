# Cotton Fabric Defect Detection System

This project is a **real-time fabric defect detection system** using OpenCV, Gradio, and machine learning (SVM with handcrafted features like LBP and Gabor filters). It's designed for integration with a camera-equipped conveyor belt to identify textile defects like **holes, tears, and stains**.

> Developed by Group 1 – 12 STEM 8B, Far Eastern University High School Inc.  
> Built with references to GitHub contributors: `x-Ck-x`, `Prahmodh-Raj1`, and `tirthajyoti`.

---

## Features

- Real-time webcam-based defect detection
- Uses:
  - **Local Binary Patterns (LBP)**
  - **Gabor Filters**
  - **Raw pixel values**
- **SVM-based classifier** trained on extracted features
- **Audio alert** on new defects
- **Defect summary counter**
- GUI built with **Streamlit**

---

## Dataset Folder Setup

Before training or using your own data, make sure to create a dataset folder:

```bash
mkdir dataset
```

Place your training images into subfolders for each class inside the `dataset/` directory, like this:

```
dataset/
├── hole/
├── tear/
├── stain/
└── no_defect/
```

Each subfolder should contain images representing that defect type.

---

## Requirements

Install the dependencies using `pip`:

```bash
pip install -r requirements.txt
```

---

## How to Run

1. Ensure your camera is connected and working.
2. Place `svm_fabric_defect.pkl`, `scaler.pkl`, and `alert.mp3` in the root directory.
3. Run the application:

```bash
python your_script_name.py
```

4. The Gradio UI will launch in your browser.
5. Select your camera, click **Start Detection**, and observe predictions in real time.

---

## File Structure

```plaintext
├── dataset/                     # Folder containing training data
│   ├── hole/
│   ├── tear/
│   ├── stain/
│   └── no_defect/
├── svm_fabric_defect.pkl       # Trained SVM model
├── scaler.pkl                  # Feature scaler
├── alert.mp3                   # Sound played upon defect detection
├── main.py                     # Main script with Gradio UI
└── README.md                   # Project documentation
```

---

## Defects Detected

- **Hole**
- **Tear**
- **Stain**
- **No defect** (default)

Each detection is buffered and stabilized using **weighted majority voting** for reliability.

---

## Camera Note

- The script auto-detects available webcams.
- If only one is connected, it defaults to index `"0"`.

---

## Resetting Detection

Click **Stop Detection** to:
- Halt the webcam feed
- Reset all internal counters

---

## Alert System

An alert sound is played only when:
- A new type of defect is detected, **or**
- More than 4 seconds have passed since the last alert

---

## Acknowledgments

This project was inspired and adapted from open-source work by:
- [x-Ck-x](https://github.com/x-Ck-x)
- [Prahmodh-Raj1](https://github.com/Prahmodh-Raj1)
- [tirthajyoti](https://github.com/tirthajyoti)

This README file was made with AI.

---

## 📄 License

For academic and prototype use only. Contact the authors for further usage.
