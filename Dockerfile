# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install dependencies (including Meta's OpenEnv directly from GitHub)
RUN pip install --no-cache-dir fastapi uvicorn pydantic requests
RUN pip install git+https://github.com/meta-pytorch/OpenEnv.git

# Copy your specific environment files into the container
# This assumes the grader builds the container from the OpenEnv root folder
COPY ./envs/free_guy /app/envs/free_guy

# Expose the API port
EXPOSE 8000

# Command to start your FastAPI server when the container boots
CMD ["uvicorn", "envs.free_guy.server.app:app", "--host", "0.0.0.0", "--port", "8000"]