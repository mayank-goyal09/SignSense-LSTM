# 🤟 SignSense-LSTM

<div align="center">
  <img src="https://img.shields.io/badge/Status-Live-00ff88?style=for-the-badge&logo=statuspage&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/AI-TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
</div>

<br />

<div align="center">
  <svg width="400" height="100" viewBox="0 0 400 100" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect width="400" height="100" rx="20" fill="#07070a"/>
    <text x="50%" y="35%" dominant-baseline="middle" text-anchor="middle" fill="#00ff88" font-family="Arial" font-size="18" font-weight="bold">Real-Time Sign Language Translation</text>
    <text x="50%" y="60%" dominant-baseline="middle" text-anchor="middle" fill="#ffffff" font-family="Arial" font-size="14">AI-Powered Motion Sequence Recognition</text>
    <text x="50%" y="85%" dominant-baseline="middle" text-anchor="middle" fill="rgba(255,255,255,0.4)" font-family="Arial" font-size="12">Bridging the Gap with LSTM Neural Networks</text>
    <rect x="10" y="10" width="380" height="80" rx="15" stroke="#00ff88" stroke-opacity="0.1" stroke-width="2"/>
  </svg>
</div>

---

### 📝 Brief Summary
We developed **SignSense-LSTM**, a real-time sign language translation system designed to bridge the communication gap for the Deaf and Hard-of-Hearing community. Unlike traditional static gesture recognizers, our solution focuses on temporal motion by leveraging **MediaPipe** to extract 21 precise hand landmark coordinates ($x, y, z$) and feeding 30-frame sequences into a **Long Short-Term Memory (LSTM)** neural network. 

This architecture allows the model to understand the flow of a sign over time, achieving **92% classification accuracy** for dynamic gestures. Built using Python, TensorFlow, and OpenCV, the system processes live video feeds to overlay translated text instantly, providing a lightweight, privacy-conscious, and hardware-efficient tool for real-world accessibility.

---

### 🚀 Live Deployment
The application is deployed on **Streamlit Community Cloud** and is optimized for high-performance browser-based inference.

**Live Application:** [https://signsense-lstm-project.streamlit.app/](https://signsense-lstm-project.streamlit.app/)

#### 💻 Local Development
To run this project locally, simply follow these steps:
```bash
# Clone the repository
git clone https://github.com/mayank-goyal09/SignSense-LSTM.git

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run streamlit_app.py
```

---

### 🔍 Project at a Glance

| 🏆 Key Technical Achievements | ⚡ System Highlights |
| :--- | :--- |
| **Temporal Awareness**: Implemented LSTM layers to recognize gestures spanning multiple seconds, not just static frames. | **Zero-Latency Inference**: Optimized the model to provide real-time translations directly in the browser feed. |
| **High Precision Tracking**: Utilized MediaPipe Landmarks for stable 3D hand tracking even in variable lighting. | **Hardware Efficient**: Designed to run smoothly on standard laptops without requiring a dedicated GPU. |
| **92% Accuracy**: Trained on custom datasets to ensure robust recognition of conversational sign language fragments. | **Privacy First**: Secure local frame processing—your video feed is never uploaded to any cloud server. |

---

### 🛠️ Technology Stack

| Category | Technology | Role in Ecosystem |
| :--- | :--- | :--- |
| **Core Engine** | ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white) | Primary programming language and logic handler. |
| **Deep Learning** | ![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) | Used for building and training the LSTM Neural Network. |
| **Computer Vision** | ![OpenCV](https://img.shields.io/badge/-OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white) | Used for real-time video capture and frame pre-processing. |
| **Hand Tracking** | ![MediaPipe](https://img.shields.io/badge/-MediaPipe-00bfff?style=flat-square&logo=google-chrome&logoColor=white) | Extracts 21 3D hand landmarks for model input. |
| **Web UI** | ![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) | Provides the premium dark-themed user interface. |

---

### 🔄 Project Workflow

1.  **Capture**: Live video feed is processed frame-by-frame via OpenCV.
2.  **Extract**: MediaPipe extracts hand landmarks, converting visual data into a spatial coordinate array.
3.  **Sequence**: The system buffers 30 consecutive frames of coordinates to form a "motion signature."
4.  **Predict**: The LSTM model analyzes the 30-frame sequence to classify the dynamic gesture.
5.  **Translate**: The result is rendered instantly on the Glassmorphic Streamlit Dashboard.

---

### ⚠️ Challenges & Difficulties Faced

Building a production-ready DL app on Streamlit Cloud presented several technical hurdles:

*   **Linux Environment Conflicts**: Standard OpenCV packages require GUI system libraries (`libGL.so.1`) which are missing on headless servers. We solved this by implementing a `packages.txt` with specific `apt-get` dependencies (`libgl1-mesa-glx`, `libglib2.0-dev`).
*   **Python Versioning**: Python 3.14 (pre-release) had no stable TensorFlow wheels. We successfully locked the environment to **Python 3.12** using Streamlit Cloud's advanced settings to ensure stability.
*   **Dependency Shadowing**: Encountered a critical `ImportError` where a local `utils.py` clashed with OpenCV's internal modules. We refactored the entire project to use `lm_utils.py` to eliminate name shadowing.
*   **Native C-Binding Errors**: MediaPipe's Tasks API failed on specific cloud architectures. We implemented a robust **Legacy API Fallback** system to ensure the hand landmarker works across all environments.

---

### ✨ Premium User Experience

The **SignSense** interface was designed to feel premium, modern, and high-performance:

*   **Glassmorphism Design**: High-end UI panels with `backdrop-filter` blur effects and translucent borders.
*   **Neon Aesthetics**: Dark mode optimized with vibrant neon green (`#00ff88`) and deep violet accents.
*   **Micro-Animations**: Custom CSS animations for the hero banner, pulse effects on the camera feed, and smooth loading transitions.
*   **Live Metrics Dashboard**: Real-time probability bars showing the neural network's confidence for every gesture.
*   **Translation History**: A searchable log of previous detections to help users track conversational strings.

---

### 📖 How to Use

1.  **Grant Permissions**: Allow your browser to access the camera when prompted.
2.  **Start Engine**: Click the **"▶ Start Engine"** button in the sidebar.
3.  **Perform Sign**: Move your hand clearly within the frame. (Try "Hello" or "How are you").
4.  **Observe**: Watch the "Current Detection" card update in real-time as the LSTM processes your movement.
5.  **History**: Scroll down to see your translation sequence in the history log.

---
<div align="center">
  <sub>Developed for Accessibility • Powered by AI • Built for the Community</sub>
</div>
