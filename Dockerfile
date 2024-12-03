# Use the official Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy project files to the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run user_conversion.py to generate model files
RUN python user_conversion.py

# Expose port 8080
EXPOSE 8080

# Start the Flask application
CMD ["python", "app.py"]
