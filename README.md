## 📽️ Violence Detection Demo

[![GitHub stars](https://img.shields.io/github/stars/null-void-Q/real-time-violence-detection.svg?style=social)](https://github.com/null-void-Q/real-time-violence-detection/stargazers)
[![Python version](https://img.shields.io/badge/python-3.13%2B-brightgreen.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/Powered%20by-Gradio-orange.svg)](https://gradio.app)
![Docker](https://img.shields.io/badge/Docker-%230DB7ED.svg?style=for-the-badge&logo=docker&logoColor=white)

A lightweight demo that uses a **DeepMind [Kinetics‑I3D](https://github.com/deepmind/kinetics-i3d "Kinetics‑I3D")** backbone, built through end-to-end pipeline: **data collection** of violent/non-violent video snippets from public and private sources, **dataset curation and labeling** (35k clips, 64 frames each), followed by **fine-tuning** the I3D model on this custom dataset, and finally deployed via a Gradio UI to flag potentially violent content in real time.  

- **Purpose:** Show how an I3D model can be repurposed for real-time violence detection – not a production-ready system.  
- **Performance (limited test set):** 91.6% accuracy, 89.6% precision, 84.7% recall.  

*Note:* The model is built on an older checkpoint and a private dataset with limited diversity and size, so treat it as a proof-of-concept only. Use it to explore the workflow or as a baseline for further research.

<div align="center">
  <a href="https://github.com/user-attachments/assets/dd25bdec-b188-4eb7-a1fe-5cc70b280389">
    <img src="https://github.com/user-attachments/assets/dd25bdec-b188-4eb7-a1fe-5cc70b280389" alt="Interface" width="800">
  </a>
</div>

## Running the Code
> [!WARNING]
> Tested on NVIDIA GPUs with CUDA support, CPU performance is not fully optimized or thoroughly tested.

> [!NOTE]
> The browser will block access to the webcam unless the project is accessed via localhost or 127.0.0.1.

### Docker
#### Prerequisites
- ✅ [Docker](https://docs.docker.com/get-docker/) installed
- ✅ [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support

1. Clone the Repository
```bash
git clone https://github.com/null-void-Q/real-time-violence-detection.git
cd real-time-violence-detection
```

2. Build and Run with Docker Compose
```bash
docker compose up
```
or
```bash
docker-compose up
```

The application will start on:
> **http://127.0.0.1:7860**

> Head to [localhost:7860](http://127.0.0.1:7860) in your browser to use the web interface

---

### Python
#### Prerequisites
- ✅ [Python 3.13](https://www.python.org/downloads/) installed
- ✅ [FFmpeg](https://ffmpeg.org/download.html) installed


1. Clone the Repository
```bash
git clone https://github.com/null-void-Q/real-time-violence-detection.git
cd real-time-violence-detection
```

2. Create and Activate a Virtual Environment (Recommended)

```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate it:
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

3. Install Dependencies
Install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

4. Run the Demo
Start the server with:

```bash
python app.py
```

The application will start on:
> **http://127.0.0.1:7860**

> Head to [localhost:7860](http://127.0.0.1:7860) in your browser to use the web interface

---
## TODO
- ☐ Improve accuracy of measured processing metrics
- ☐ Clarify Hardware requirements and expected performance
- ☐ Improve model performance across diverse contexts
