# Concurrent Multimodal Ingestion Pipeline (CMIP) with VLM Watchman Daemon

An asynchronous, high-throughput media ingestion engine designed to optimize preprocessing pipelines for Multimodal AI applications. The system leverages an event-driven directory observer to ingest raw graphics, offloads heavy CPU/GPU matrix transformations to structured thread pools, and generates automated image captions utilizing an integrated Hugging Face Vision-Language Model (VLM).

## 🚀 Core Features & Architecture

- **Event-Driven Hot Folder Ingestion:** Leverages a custom FileSystemEventHandler daemon to observe raw directory buffers seamlessly.
- **Asynchronous Telemetry & IO Integration:** Operates on an unblocked `asyncio` main loop orchestration layer to decouple logging operations from compute tasks.
- **Multi-Threaded Hardware Optimization:** Dispatches resource-intensive OpenCV resizing and PyTorch matrix processing pipelines directly into isolated `ThreadPoolExecutor` cycles.
- **Multimodal AI Enrichment:** Uses the Salesforce BLIP Vision-Language Model via Hugging Face to run real-time contextual inference on inbound files.

## 🛠️ Tech Stack & Prerequisites

- **Language:** Python 3.10+
- **Concurrency Frameworks:** `asyncio`, `concurrent.futures`, `threading`
- **Core AI Ecosystem:** `torch` (PyTorch), `transformers` (Hugging Face BLIP VLM)
- **Computer Vision Infrastructure:** `opencv-python` (OpenCV Matrix Operations), `pillow` (PIL Overlays)
- **OS File Monitor:** `watchdog`

## 📦 Local Installation & Deployment

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Concurrent-Multimodal-Ingestion-Pipeline.git](https://github.com/YOUR_USERNAME/Concurrent-Multimodal-Ingestion-Pipeline.git)
   cd Concurrent-Multimodal-Ingestion-Pipeline

2. **Establish Environment & Core Dependencies:**
   ```bash
   python -m venv cmip_env
   cmip_env\Scripts\activate.bat

   pip install -r requirements.txt
    ```
4. **Initialize the Interactive Watchman Engine:**
    ```bash
   python src/main_watchman.py
    ```

6. **Execute CLI Flow:**
   - Type start to spin up the processing loop.
   - Drop images (.jpg, .png) inside data/raw/.
   - View your production-ready, AI-captioned, watermarked results in data/processed/.
