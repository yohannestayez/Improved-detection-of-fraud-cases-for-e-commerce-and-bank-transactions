# Use a Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the Flask app and models
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Expose the API port
EXPOSE 5000

# Run the Flask application
CMD ["python", "serve_model.py"]
