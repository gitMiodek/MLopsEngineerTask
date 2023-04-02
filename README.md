
# ML API for Plane/Car Recognition

## ML Model
This project is a simple machine learning model that classifies images as either a plane or a car. It uses a pre-trained ResNet18 model with the last layer adjusted to perform binary classification.

## REST API Endpoint
The ML model is wrapped in a REST API that accepts a URL to an image in JSON format. The API returns a list of probabilities for each class (plane and car) and the predicted label for the image.
- API documentation is in JSON format in openapi.json

## Dockerized Application 
The entire application has been containerized using Docker Compose. It consists of two services:

- The first service runs the REST API.
- The second service runs tests to ensure the proper functionality of the API. The testing service runs for approximately 30 seconds after the API service.

## Continuous Integration Pipeline
The project includes a working CI pipeline that runs the Docker Compose file. If the tests are successful, the Docker container exits with a code of 0 and the image containing the API service is pushed to Docker Hub.


# Instruction to use API
0. Make sure there is docker installed on your local machine
1. Clone git repository
2. Run command: docker build -t <api_name> . (it will build an image based on docker file)
3. Run command: docker run -p 8000:8000 <api_name> (it will run api)

# Instuction to perfom testing
1. Run command docker-compose up --build --exit-code-from tests


