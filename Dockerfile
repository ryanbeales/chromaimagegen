FROM ubuntu:24.10

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*

# Install python
RUN uv python install 3.12

# Otimize build time
COPY ./app/.python-version /app/.python-version
COPY ./app/uv.lock /app/uv.lock
COPY ./app/pyproject.toml /app/pyproject.toml

# Sync proejct dependencies
RUN uv sync --frozen


# Copy the rest of the project into the image
ADD ./app/ /app

CMD ["uv", "run", "uvicorn", "main:api", "--host", "0.0.0.0", "--port", "8000"]