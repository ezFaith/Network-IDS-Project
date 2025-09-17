# Use a lightweight Python base image with a version compatible with your dependencies
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy all your project files into the container
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Define the command to run your app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]