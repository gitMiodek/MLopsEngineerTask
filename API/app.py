import urllib.request
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torchvision import transforms
from cnn import model
from PIL import Image
from typing import List
import uvicorn

# Create an instance of api
app = FastAPI()

# Img preprocess function
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# Initialize pre trained model
plane_v_car = model


# Define input schema
class InputData(BaseModel):
    """
    Input data is a JSON with a url address of the image
    """
    image_url: str


# Define output schema
class OutputResponse(BaseModel):
    """
    Output Response is a JSON with a list of probabilities
    """
    probabilities: List[float]
    predicted_label: str


print("helo")
@app.get("/check")
async def get_checked():
    return {"hello":"world"}

@app.post("/predict", response_model=OutputResponse)
async def img_prediction(img_data: InputData):
    """
       Endpoint to predict class probabilities for an image given its URL.

       Args:
           - InputData: input data containing the image URL.

       Returns:
           - Output: class probabilities for classes [dog, cat] and label for the predicted class
       """
    # Get image url from the json
    img_url = img_data.image_url
    # img_url = data.get("image_url")

    # Check if there is an url
    if not img_url:
        raise HTTPException(status_code=422, detail=[
            {"loc": ["body", "image_url"], "msg": "field required", "type": "value_error.missing"}])
    try:
        # Download image from URL
        urllib.request.urlretrieve(img_url, "sample_image.jpg")
        # Preprocess the image
        preprocessed_image = transform(Image.open("sample_image.jpg").convert("RGB")).unsqueeze(0)
        # Feed data into a model
        plane_v_car.eval()
        with torch.no_grad():
            # Get output
            output = plane_v_car(preprocessed_image)
            # Calculate class probabilities distributed over 2 classes
            probabilities = torch.nn.functional.softmax(output, dim=1)
            probabilities_list = probabilities[0].tolist()

            if probabilities_list[0] > probabilities_list[1]:
                label = "Plane"
            else:
                label = "Car"
            return {"probabilities": probabilities_list,
                    "predicted_label": label}
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=str(e))
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)