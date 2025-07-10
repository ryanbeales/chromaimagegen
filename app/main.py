# API includes
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from asyncio import Lock


# Request object and settings
from pydantic import BaseModel
from pydantic_settings import BaseSettings

# File handling and temporary directories
import os
import uuid

# Generate the seed for image generation
import random

# Image generation libraries
import torch
from diffusers import ChromaPipeline

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

# This class allows multiple requests to make use of the same shared pipeline instance
class SharedPipeline:
    pipeline: ChromaPipeline = None
    device: str = None

    def start(self, settings: Settings = Settings()):
        # Auto detect cuda or mps, fail with anything else (untested because I don't have that hardware).
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps" # For Apple Silicon Macs
        else:
            raise Exception("No suitable device found. Please ensure you have a compatible GPU or MPS support.")

        # Create the pipeline, will automatically download the model if not cached already, and will load layers in to the GPU. device_map="balanced" will 
        # try to fill GPU memory first then use CPU or disk after.
        self.pipeline = ChromaPipeline.from_pretrained(settings.model, torch_dtype=torch.bfloat16, device_map="balanced", cache_dir=settings.model_cache_dir)

    async def generate(self, prompt: str, negative_prompt: str = "", width: int = 512, height: int = 512, num_inference_steps: int = 15):
        if not self.pipeline:
            raise Exception("Pipeline is not initialized. Please call start() first.")
    
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

# Create the image pipeline on fastapi startup
@api.on_event("startup")
def startup_event():
    try:
        shared_pipeline.start()
    except Exception as e:
        raise e

@api.get("/")
def get_root():
    return {"message": "Welcome to the Image Generation Server!"}

# Endpoint to generate an image based on the request
@api.post("/generate-image")
async def generate_image(request: ImageRequest):
    try:
        # Check if the pipeline has started
        if not shared_pipeline.pipeline:
            raise HTTPException(status_code=500, detail="Image generation pipeline is not initialized.")
        
        # Run the image generation in a separate thread to avoid blocking the fastapi event loop.
        output = await run_in_threadpool(
            lambda: shared_pipeline.pipeline(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                num_inference_steps=15
            )
        )   

        # Save the generated image to disk
        image = output.images[0]
        image_path = os.path.join("/images", f"{uuid.uuid4()}.png")
        image.save(image_path)

        # Return the image location and seed value in the response
        return {"image_url": image_path}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        elif hasattr(e, 'message'):
            raise HTTPException(status_code=500, detail=e.message)
        raise HTTPException(status_code=500, detail=str(e))

# Start uvicorn server if this script is run directly
if __name__ == "__main__":
    uvicorn.run("app.main:api", host="127.0.0.1", port=8000, reload=True)