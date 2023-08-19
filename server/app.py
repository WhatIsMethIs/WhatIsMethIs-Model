from flask import Flask, request
import json
import torch
from torchvision import transforms
from PIL import Image
from model.pill_model import PillModel
import configparser

app = Flask(__name__)

config_path = "./model/modeling_config.ini"
label_to_index_path = "./model/230819_linear764_764_200_PyTorchModel_dict.json"
model_path = "./model/230819_linear764_764_200_PyTorchModel.pt"

config = configparser.ConfigParser()
config.read(config_path, encoding="UTF-8")

index_to_label = None
with open(label_to_index_path) as json_file:
    index_to_label = {value: key for key, value in json.load(json_file).items()}

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PillModel(config["PT_model_info"]).to(device)

checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def transform_image(infile):
    image = Image.open(infile)
    print(image)

    image_size = int(config["PT_model_info"]["input_dim"])
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    image = transform(image)
    image = image[None, :]

    return image


def get_prediction(input_tensor):
    outputs = model.forward(input_tensor)
    _, y_hat = outputs.max(1)
    prediction = y_hat.item()
    return prediction


def render_prediction(prediction_idx):
    return index_to_label[prediction_idx]


@app.route("/medicines/identify", methods=["POST"])
def identifyMedicines():
    image_file = request.files["file"]

    input_tensor = transform_image(image_file)
    prediction_idx = get_prediction(input_tensor)
    class_id = render_prediction(prediction_idx)

    return {"class_id": class_id}


if __name__ == "__main__":
    app.run(port=5000)
