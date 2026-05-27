import asyncio
import concurrent.futures
import time
import threading
from pathlib import Path
import cv2
from PIL import Image, ImageDraw

# --- HUGGING FACE AI IMPORTS ---
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# --- DIRECTORY MONITORING IMPORTS ---
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class PipelineEventHandler(FileSystemEventHandler):
    """Listens for file drops, but only processes them if the Watchman is AWAKE."""
    def __init__(self, main_loop, executor, engine):
        self.main_loop = main_loop
        self.executor = executor
        self.engine = engine

    def on_created(self, event):
        if not self.engine.is_awake:
            return 

        if not event.is_directory:
            file_path = Path(event.src_path)
            if file_path.suffix.lower() in {'.jpg', '.png', '.jpeg'}:
                print(f"\n[Watchman] New file spotted: {file_path.name}")
                # Dispatch task to the background async loop safely
                asyncio.run_coroutine_threadsafe(
                    self.dispatch_processing(file_path), self.main_loop
                )

    async def dispatch_processing(self, file_path: Path):
        await asyncio.sleep(0.5)  # Short pause to let OS finish writing image to disk
        result_log = await self.main_loop.run_in_executor(
            self.executor, 
            self.engine.process_image_heavy_compute, 
            file_path
        )
        await self.engine.async_telemetry_logger(result_log)


class WatchmanAIPipeline:
    def __init__(self, input_dir: str, output_dir: str, max_workers: int = 4):
        self.input_path = Path(input_dir)
        self.output_path = Path(output_dir) 
        self.max_workers = max_workers
        self.is_awake = False  
        
        # Self-healing path creation
        self.input_path.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # --- LIGHT UP THE AI BRAIN ---
        print("\n[SYSTEM] Loading AI Brain (Salesforce BLIP VLM via Hugging Face)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        print(f"[SYSTEM] AI Brain operational on device: {self.device.upper()}.\n")

    def process_image_heavy_compute(self, image_path: Path) -> str:
        """CPU/GPU-Bound Task: Standardizes image and infers context via VLM."""
        try:
            # 1. Media Manipulation via OpenCV
            img_bgr = cv2.imread(str(image_path))
            if img_bgr is None:
                return f"ERROR: Unreadable image file {image_path.name}"
            
            img_resized = cv2.resize(img_bgr, (960, 540), interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # 2. Multimodal AI Inference
            ai_inputs = self.processor(img_pil, return_tensors="pt").to(self.device)
            out = self.model.generate(**ai_inputs, max_new_tokens=20)
            caption = self.processor.decode(out[0], skip_special_tokens=True).title()

            # 3. Pillow UI Stamp
            draw = ImageDraw.Draw(img_pil)
            draw.rectangle([10, 490, 950, 530], fill=(0, 0, 0)) # Clean black banner
            
            watermark_text = f"CAPTION: {caption}"
            draw.text((25, 505), watermark_text, fill=(255, 255, 255))
            
            # 4. Production Ready Export
            output_file = self.output_path / f"ai_secured_{image_path.name}"
            img_pil.save(output_file, "JPEG", quality=85)
            
            # Delete original from raw folder to keep input clean
            image_path.unlink() 
            return f"Success: Captured metadata for '{image_path.name}'"
        except Exception as e:
            return f"ERROR processing {image_path.name} : {str(e)}"
    
    async def async_telemetry_logger(self, status_message: str):
        """Non-blocking tracker log wrapper."""
        await asyncio.sleep(0.01)
        print(f"[Telemetry] {status_message}")
        print("ai_watchman> ", end="", flush=True)


def start_background_async_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


if __name__ == "__main__":
    # Pointing seamlessly to your newly segregated directory structure
    INPUT_WORKSPACE = r"E:\Projects & skills\Concurrent Multimodal Ingestion Pipeline (CMIP)\data\raw"
    OUTPUT_WORKSPACE = r"E:\Projects & skills\Concurrent Multimodal Ingestion Pipeline (CMIP)\data\processed"

    engine = WatchmanAIPipeline(input_dir=INPUT_WORKSPACE, output_dir=OUTPUT_WORKSPACE)
    
    # Run the Asyncio Traffic Director in a daemonized background thread
    background_loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=start_background_async_loop, args=(background_loop,), daemon=True)
    loop_thread.start()

    # Wire up the ThreadPool for heavy tensor computing
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=engine.max_workers)
    event_handler = PipelineEventHandler(background_loop, executor, engine)
    
    observer = Observer()
    observer.schedule(event_handler, str(engine.input_path), recursive=False)
    observer.start()

    print("==========================================================")
    print("      AI WATCHMAN ONLINE (Status: ASLEEP)")
    print("==========================================================")
    print(" Console Commands Switch:")
    print("  'start' -> Wake watchman (Processes & captions drops)")
    print("  'stop'  -> Stand down watchman (Ignores drops)")
    print("  'exit'  -> Kill engine daemon safely\n")

    try:
        while True:
            cmd = input("ai_watchman> ").strip().lower()
            if cmd == "start":
                engine.is_awake = True
                print("[SYSTEM] Watchman is actively guarding. Drop your images into 'data/raw'...")
            elif cmd == "stop":
                engine.is_awake = False
                print("[SYSTEM] Watchman resting. Ingestion paused.")
            elif cmd == "exit" or cmd == "quit":
                print("[SYSTEM] Halting thread cycles...")
                break
    except KeyboardInterrupt:
        print("\n[SYSTEM] Interrupt detected.")
    finally:
        engine.is_awake = False
        observer.stop()
        observer.join()
        background_loop.call_soon_threadsafe(background_loop.stop)
        executor.shutdown(wait=False)
        print("[SYSTEM] Pipeline successfully terminated.")