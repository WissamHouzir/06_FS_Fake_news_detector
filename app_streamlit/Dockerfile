# Use a lightweight base image with Miniconda preinstalled
FROM continuumio/miniconda3

# Install useful system tools (needed for many Python libs)
RUN apt-get update && apt-get install -y \
    nano \
    unzip \
    curl \
 && rm -rf /var/lib/apt/lists/*

# --- HuggingFace best practices ---
# Create a non-root user (avoid running as root in containers)
RUN useradd -m -u 1000 user
USER user

# Define environment variables
# - HOME: user home directory
# - PATH: add user local bin to PATH (so pip-installed binaries are found)
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory (all code will live here)
WORKDIR $HOME/app

COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files into the container (make sure user owns them)
COPY --chown=user . $HOME/app

# Expose the port used by Streamlit / HuggingFace Spaces 
EXPOSE 7860

# Start Streamlit when the container runs
CMD ["streamlit", "run", "--server.port=7860", "app.py"]