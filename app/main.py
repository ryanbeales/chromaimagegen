# API includes
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from asyncio import Lock, sleep, create_task
from starlette.responses import FileResponse 

# Request object and settings
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

# File handling and temporary directories
import os
import uuid

# Generate the seed for image generation
import random

# Image generation libraries
import torch
from diffusers import ChromaPipeline

import gc

import logging
logger = logging.getLogger(__name__)

# Pydantic settings class, which allows overriding of values via environment variables
class Settings(BaseSettings):
    # Default location for the model cache directory. These models are large. For the default model here expect to have around 19GB of free space.
    # If you _don't_ set this to a permanent location (eg: volume mount in docker), it will try and download the models each time you start the server.
    model_cache_dir: str = "/model_cache" # Default location for the model cache
    # Output location for generated images.
    image_dir: str = "/image_dir"
    # Default model to use for image generation. Chroma is uncensored and can generate a wide variety of images. Expect to have about 19GB of GPU memory
    # available. On a 4090 image generation is about 7 seconds per image and all layers will still not fit on the GPU.. On a 3060 (after 4bit quantizing),
    # ~3 minutes. On an m3 ~5 minutes.
    model: str = "lodestones/Chroma"
    # Idle timeout in seconds before the model is unloaded from GPU memory. Set to 0 to disable.
    idle_timeout: int = 1 * 60  # 1 minutes
    # Load the model on server start, or on first request
    load_model_on_start: bool = False

    model_config = SettingsConfigDict(env_file=".env") # Load settings from .env file if available

# This class allows multiple requests to make use of the same shared pipeline instance
class SharedPipeline:
    pipeline: ChromaPipeline = None
    device: str = None
    lock = Lock()  # Lock to ensure thread-safe access to the pipeline
    cleanup_task = None  # Task to handle unloading the pipeline after inactivity

    def start(self, settings: Settings = Settings()):
        logger.info("Starting the image generation pipeline...")
        # Auto detect cuda or mps, fail with anything else (untested because I don't have that hardware).
        if torch.cuda.is_available():
            logger.info("Using CUDA for image generation.")
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            logger.info("Using MPS for image generation (Apple Silicon)")
            self.device = "mps" # For Apple Silicon Macs
        else:
            logger.error("No suitable device found for image generation. Please ensure you have a compatible GPU or MPS support.")
            raise Exception("No suitable device found. Please ensure you have a compatible GPU or MPS support.")

        # Create the pipeline, will automatically download the model if not cached already, and will load layers in to the GPU. device_map="balanced" will 
        # try to fill GPU memory first then use CPU or disk after.
        self.pipeline = ChromaPipeline.from_pretrained(settings.model, torch_dtype=torch.bfloat16, device_map="balanced", cache_dir=settings.model_cache_dir)
        logger.info(f"Pipeline loaded successfully with model: {settings.model}")

    def unload(self):
        if self.pipeline:
            logger.info("Unloading the image generation pipeline to free up GPU memory")
            # Unload the pipeline from memory
            del self.pipeline
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("Pipeline unloaded successfully.")

    def generate(self, prompt: str, negative_prompt: str = "", width: int = 512, height: int = 512, num_inference_steps: int = 15):
        # The model may have been loaded if the server was started with `load_model_on_start`, but if not, we need to ensure it is loaded before generating an image.
        if not self.pipeline:
            self.start()
            
        # Generate the image using the pipeline
        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps
        )
        return output

# Request object for image generation
class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = "low quality, bad anatomy, extra digits, missing digits, extra limbs, missing limbs, mutated hands and fingers, poorly drawn hands and fingers, poorly drawn face, mutation, deformed, blurry, out of focus, long neck, long body, long arms, long legs, long fingers, long toes"
    width: int = 512
    height: int = 512

# Initialize settings, API, and shared pipeline, mount the output directory as static files in fastapi.
settings = Settings()
api = FastAPI()
shared_pipeline = SharedPipeline()
api.mount("/images", StaticFiles(directory=settings.image_dir), name="images")

@api.get("/")
def get_root():
    return FileResponse("ui/image_generator.html")

@api.get("/health")
def get_health():
    return {"status": "ok"}

# Create the image pipeline on fastapi startup (if configured)
@api.on_event("startup")
def startup_event():
    if settings.load_model_on_start:
        logger.info("Loading the image generation pipeline on startup")
        try:
            shared_pipeline.start()
        except Exception as e:
            raise e
        raise e

# Unload the pipeline after a period of inactivity to free up GPU memory
cleanup_task = None
async def inactivity_cleanup():
    await sleep(settings.idle_timeout)
    logger.info(f"Unloading the pipeline after {settings.idle_timeout} seconds of inactivity")
    async with shared_pipeline.lock:
        if shared_pipeline.pipeline:
            # Unload the pipeline if it has been idle for the specified timeout
            shared_pipeline.unload()

# Endpoint to generate an image based on the request
@api.post("/generate-image")
async def generate_image(request: ImageRequest):
    try:
        async with shared_pipeline.lock:
            # Access global cleanup_task variable
            global cleanup_task
            
            # Cancel any existing cleanup task
            if cleanup_task and settings.idle_timeout != 0:
                try:
                    cleanup_task.cancel()
                except Exception as e:
                    logger.warning(f"Failed to cancel cleanup task: {e}")

            # Run the image generation in a separate thread to avoid blocking the fastapi event loop.
            output = await run_in_threadpool(
                lambda: shared_pipeline.generate(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    width=request.width,
                    height=request.height,
                    num_inference_steps=15
                )
            )   

            # Start a new cleanup task to unload the pipeline after inactivity
            if settings.idle_timeout != 0:
                cleanup_task = create_task(inactivity_cleanup())

        # Save the generated image to disk
        image = output.images[0]
        image_filename = f"{uuid.uuid4()}.png"
        image_path = os.path.join(settings.image_dir, image_filename)
        image.save(image_path)

        # Return the image location and seed value in the response
        return {"image_url": f"/images/{image_filename}"}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        elif hasattr(e, 'message'):
            raise HTTPException(status_code=500, detail=e.message)
        raise HTTPException(status_code=500, detail=str(e))

# Start uvicorn server if this script is run directly
if __name__ == "__main__":
    uvicorn.run("app.main:api", host="127.0.0.1", port=8000, reload=True)