# Use a lightweight Python image
FROM python:3.10-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory inside the container
WORKDIR /app

# Install dependencies 
RUN pip install --no-cache-dir fastapi uvicorn pydantic requests
RUN pip install git+https://github.com/meta-pytorch/OpenEnv.git

# Copy your specific environment files into the container
# Use --chown=user to ensure the non-root user has permission to read the files
COPY --chown=user ./envs/free_guy /app/envs/free_guy

# Expose the Hugging Face required port
EXPOSE 7860

# Command to start your FastAPI server on port 7860
CMD ["uvicorn", "envs.free_guy.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
