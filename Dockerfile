# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

RUN apt-get update && apt-get install -y tesseract-ocr

# Copy the current directory contents into the container at /app
COPY . /app

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # Technically, the index-url here is redundant but good for clarity in the file.

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose ports for the web app and API
EXPOSE 8501 8080

# Start both the web app and API
CMD ["sh", "-c", "streamlit run src/web_app/app_ollama.py --server.port 8501 & python src/api/server.py"]