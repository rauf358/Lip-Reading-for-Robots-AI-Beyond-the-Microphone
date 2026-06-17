## 🛠 Tech Stack

This project combines **real-time computer vision, geometric reasoning, and efficient landmark tracking** to build a lightweight visual speaker detection system capable of running on standard hardware with minimal latency.

### Core Technologies

### 🐍 Python

Used as the primary development language for rapid prototyping and real-time system integration.

### 📷 OpenCV

Used for the complete video-processing pipeline:

* Real-time webcam feed acquisition
* Frame preprocessing
* Rendering visual overlays
* Drawing speaker bounding boxes and status labels
* Live output visualization

OpenCV served as the backbone for low-latency frame handling and visualization.

### 🧠 MediaPipe Face Mesh

The core intelligence behind the system.

Used for:

* High-speed face tracking
* Multi-face detection (**up to 5 faces simultaneously**)
* Extraction of **468 highly precise 3D facial landmarks per face**
* Accurate localization of lip landmarks for speech inference

This enabled robust facial analysis while maintaining real-time performance.

### 🔢 NumPy

Used for efficient numerical computation:

* Lip coordinate averaging
* Distance calculations
* Dynamic threshold scaling
* Frame-wise geometric analysis

---

## ⚡ Core Engineering Concepts

Beyond libraries, this project demonstrates practical application of several advanced engineering concepts:

### Real-Time Computer Vision

Processing and analyzing video streams frame-by-frame with minimal delay.

### Facial Landmark Analysis

Using dense landmark mapping to infer meaningful human behavior from subtle facial movements.

### Geometric Speech Inference

Instead of relying on microphones or audio processing, the system estimates speech activity through **visual articulation patterns**.

### Dynamic Threshold Calibration

A major engineering improvement.

Rather than using a fixed threshold, the system dynamically scales speaking sensitivity relative to face size, making detection robust across:

* Different camera distances
* Multiple users
* Varying face scales

### Multi-Person Speaker Identification

Each detected face is independently analyzed, allowing the robot to determine **who among multiple people is actively speaking**.

---

## 🧠 Technical Implementation

The system follows a lightweight yet effective inference pipeline:

### Step 1 — Face Detection & Tracking

Each incoming frame is processed using MediaPipe Face Mesh to detect and track facial landmarks in real time.

Output per face:

* 468 landmarks
* 3D positional information
* Stable tracking across frames

---

### Step 2 — Lip Landmark Extraction

Key upper and lower lip landmarks are selected to capture mouth movement accurately.

Upper lip landmarks:

* 13
* 312
* 82

Lower lip landmarks:

* 14
* 317
* 87

Using multiple points improves robustness against noise and facial variation.

---

### Step 3 — Lip Movement Quantification

The system computes the average vertical separation between upper and lower lips:

```text
Lip Opening = Avg(Lower Lip Y) − Avg(Upper Lip Y)
```

Interpretation:

* Small opening → Silent
* Large opening → Speaking

---

### Step 4 — Adaptive Thresholding

A naive fixed threshold fails when users stand at different distances.

To solve this, the system estimates face scale using:

* Forehead landmark → 10
* Chin landmark → 152

Threshold formula:

```text
Speaking Threshold = Face Height × 0.03
```

This makes the system:

* Scale-invariant
* More reliable
* Better suited for real-world deployment

---

## 🚀 Why This Project Stands Out

This project demonstrates the ability to build **deployable AI-assisted systems**, not just train models.

Key highlights:

✔ Built a real-time vision pipeline from scratch
✔ Solved a practical human-robot interaction problem
✔ Engineered adaptive multi-user detection logic
✔ Optimized for lightweight inference on CPU
✔ Converted subtle facial motion into actionable intelligence

This reflects strong capability in:

* AI Engineering
* Computer Vision
* Applied Machine Learning
* Real-Time System Design
* Human-Centered Robotics
