# Use Miniconda as the base image
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /app

# Copy the environment file first (for better caching)
COPY environment_unified.yml /app/

# Create and activate the Conda environment only if it does not already exist
RUN if [ ! -d "$CONDA_PREFIX/envs/gen2kgbot" ]; then \
        conda env create -f environment_unified.yml && \
        conda clean --all -y; \
    else \
        echo "Conda environment gen2kgbot already exists"; \
    fi

# Ensure the environment is activated and set as default
SHELL ["conda", "run", "-n", "gen2kgbot", "/bin/bash", "-c"]

# Copy the app files
COPY . /app

# Expose the FastAPI port
EXPOSE 8000

# Run the FastAPI server using Conda's Python
CMD ["conda", "run", "-n", "gen2kgbot", "python", "-m", "app.api.dataset_forge_api"]

