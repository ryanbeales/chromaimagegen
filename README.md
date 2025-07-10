# Example Chroma image generation server
A simple example of a python image generation server using huggingface diffusers pipelines. The image generation can be slow depending on the hardware and model used. The intent for this is to be a backend service for generating images with another service in front of this to handle the long connection times required and routing to different model types.

## Setup
Install `uv` by following https://docs.astral.sh/uv/getting-started/installation/

Install python 3.12 (required for sentencepiece at time of writing)
```
uv python install 3.12
```

Install dependencies:
```
cd app && uv sync
```

Create a `app/.env` file:
```
IMAGE_DIR=image_dir
MODEL_CACHE_DIR=model_cache
```

Create the directories:
```
mkdir model_cache && mkdir image_dir
```

Start server locally (`export` for linux/macos or `set` if you're on windows):
```
uv run uvicorn main:api --host 0.0.0.0 --port 8000 --env-file=.env
```

Open the swagger UI to test at http://localhost:8000/docs

## Local testing with docker
Adjust your volume mounts depending on which system you're on (windows/macos/linux etc)
```
docker build -t imagegen .
mkdir model_cache && mkdir image_dir
docker run --gpus=all -v $(pwd):/model_cache -v $(pwd)/image_dir:/image_dir  -it --rm -p 8000:8000 imagegen
```

As above, open the swagger UI to test http://localhost:8000/docs

# Speed and space requirements
On a 4090, with a high speed internet connection things will take (roughly):
- Initial model download: ~20minutes
- Subsequent server startups: ~3 minutes
- First image generation: ~1 minute
- Subsequent image generations: ~ 7 seconds

On an M3 Mac, expect model operations to be _much_ slower, in the order of ~5 minutes per image generation.

The default model requries ~20GB of disk space and GPU memory. It can be overriden by setting the `MODEL` environment variable but this is a [ChromaPipeline](https://huggingface.co/docs/diffusers/main/en/api/pipelines/chroma) so only [Chroma](https://huggingface.co/lodestones/Chroma) models should be used.

# Issues
- Multiple requests can run at once, but all will fail. Tried adding some locking but will review this later.