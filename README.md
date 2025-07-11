# 🦿 3D Project – Pose Detection and Classification

A complete system for acquiring, processing, and classifying rehabilitative movements using RealSense camera, **MediaPipe**, **Unity 3D**, and a **Transformer-based** model.

## 📄 Link to the paper

[📄 Paper_3D_gruppo 19.pdf](./Paper_3D_gruppo%2019.pdf)

## 🎥 Short demo

<video src="Video_Interfaccia_Unity_Gruppo_19.mp4" controls width="600"></video>

---

## 🚀 Key Features

- ✅ Extraction of 3D landmarks from `.bag` videos (Intel RealSense) using **MediaPipe**
- ✅ Saving data in `.npy` format for automated use
- ✅ Classification of movements through a **PoseTransformer3D** neural network
- ✅ Workflow automation via **local server** monitoring an input folder
- ✅ User interface developed in **Unity 3D** for interaction and visual feedback

---

## 🧠 System Overview

The system guides the user through a simple and automated workflow:

1. **Launching the Unity Interface**  
   The user can choose between two options: start a **new recording** or view the **history of previous acquisitions**.

2. **Movement Recording**  
   A countdown is shown to allow the patient to position themselves correctly (side view, full limb visibility).  
   The recording lasts approximately **7 seconds**, during which the user receives **real-time visual feedback**.

3. **Processing and Classification**  
   The video is saved as `.bag` and sent to a **local backend**, which:
   - Extracts **3D landmarks** with MediaPipe
   - Classifies the movement with a Transformer model
   - Computes **joint angles** frame by frame

4. **Output**  
   The user receives:
   - The recognized gesture type
   - The maximum detected angle
   - **Personalized feedback**
   - An animated 3D representation of the correct movement

5. **History**  
   Each session is saved with **date, time, and feedback**, viewable by the same user.

---

## 📁 Project Structure

````

📦 Progetto 3D
├── App/ → Interfaccia grafica realizzata in Unity 3D
├── landmark_extraction/ → Script per estrazione landmark da video .bag
├── PoseDetector/ → Modello, training, server di classificazione
├── prediction/ → Output generati dal server (classificazioni, angoli, feedback)
├── storico/ → Archivio delle acquisizioni utente (video + dati)
├── requirements.txt → Dipendenze Python
└── .gitignore → File ignorati da Git

````

---

## ⚙️ Environment Setup

### 1. Clone the repository

```bash
git clone https://github.com/tuo-utente/progetto-riabilitazione-3d.git
````

### 2. Create a virtual environment

```bash
cd landmark_extraction
python -m venv mp_env
mp_env\Scripts\activate        # Su Windows
```

### 3. Install the main dependencies

```bash
pip install mediapipe opencv-python pyrealsense2
pip install -r ../requirements.txt
```
---

## 🧪 Usage – Training and Classification

### 1. Prepare the data in data/

Make sure the structure and labels are consistent with those used during training.

### 2. Start model training

```bash
python PoseDetector/main.py
```

### 3. Start the server for automatic classification

```bash
python PoseDetector/server.py
```

📂 The server will monitor the incoming_data/ folder and automatically classify each new .npy file added.

---

## ❗ Common Issues (MediaPipe on Windows)

### Error: `ImportError: DLL load failed while importing _framework_bindings`

**Solutions:**

* Use **64-bit Python**
* Create the virtual environment using `venv` (not `conda`)
* Upgrade pip and setuptools:

  ```bash
  python -m pip install --upgrade pip setuptools wheel

  ```
* Restart your PC after installation

### Error: `ModuleNotFoundError: No module named 'pyrealsense2'`

**Solutions:**

* Install with:

  ```bash
  pip install pyrealsense2
  ```
* Make sure the virtual environment is activated
---

## 💡 Tips

* If you use **VS Code**, select the correct Python interpreter (`mp_env`) in the bottom right corner
* Always activate the virtual environment before running scripts
* If you work in a team, you can create a separate `dev-requirements.txt` file for development tools

---


## 🧾 Credits

* User interface developed in **Unity 3D**
* 3D animated models generated with **Mixamo** and refined in **Blender**
* Classification model based on **Transformer architecture**

---

