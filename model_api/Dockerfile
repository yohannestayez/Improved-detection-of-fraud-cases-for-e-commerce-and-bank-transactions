# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install additional Python dependencies
RUN pip install -r requirements.txt

# Expose port 5000 for the Flask API
EXPOSE 5000

# Run the Flask application
CMD ["python", "serve_model.py"]
