# Choose docker image base
FROM ubuntu:22.04
# Install system dependencies
RUN apt-get update && apt-get install -y python3-pip
# Copy files into a container
COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
COPY . .
# Install dependencies from requirements file

# Set the working directory
WORKDIR /API
# Expose port
EXPOSE 8000
# Run application

CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]


