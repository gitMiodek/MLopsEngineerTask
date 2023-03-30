import requests
import json

URL = "http://localhost:8000/predict"


# test if model recognize a plane
def test_plane():
    plane_img = "https://upload.wikimedia.org/wikipedia/commons/f/fc/Tarom.b737-700.yr-bgg.arp.jpg"
    response = requests.post(url=URL, data=json.dumps({"image_url": plane_img}))

    assert response.status_code == 200
    assert response.json()['predicted_label'] == "Plane"


# test if model recognize a car
def test_car():
    car_img = "https://upload.wikimedia.org/wikipedia/commons/2/25/2015_Mazda_MX-5_ND_2.0_SKYACTIV-G_160_i-ELOOP_Rubinrot-Metallic_Vorderansicht.jpg"
    response = requests.post(url=URL, data=json.dumps({"image_url": car_img}))
    assert response.status_code == 200
    assert response.json()['predicted_label'] == "Car"
