from flask import Flask, request
import json
import torch
from torchvision import transforms
from PIL import Image
from model.pill_model import PillModel
import configparser
import cv2
import numpy as np
from preprocessing import ImageProcess

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
    preprocessor = ImageProcess()

    processed_image = preprocessor.CropShape(f"./upload/{infile}")
    processed_image = preprocessor.max_con_CLAHE(processed_image)
    processed_image = preprocessor.max_con_CLAHE(processed_image)

    cv2.imwrite(f"./upload/preprocessed_{infile}", processed_image)
    image = Image.open(f"./upload/preprocessed_{infile}")

    image_size = int(config["PT_model_info"]["input_dim"])
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    image = transform(image)
    image = image[None, :]

    return image


def get_prediction(input_tensor):
    outputs = model.forward(input_tensor)

    outputs_numpy = outputs.detach().numpy()
    outputs_exp = np.exp(outputs_numpy)
    sum_outputs_exp = np.sum(outputs_exp)
    outputs_softmaxed = outputs_exp / sum_outputs_exp

    np.set_printoptions(precision=5, suppress=True)
    print(outputs_softmaxed)

    _, y_hat = outputs.max(1)
    prediction = y_hat.item()
    return prediction


def render_prediction(prediction_idx):
    return index_to_label[prediction_idx]


@app.route("/medicines/identify", methods=["POST"])
def identifyMedicines():
    image_file = request.files["file"]
    image_file.save(f"./upload/{image_file.filename}")

    input_tensor = transform_image(image_file.filename)
    prediction_idx = get_prediction(input_tensor)
    class_id = render_prediction(prediction_idx)

    return {"class_id": class_id}


if __name__ == "__main__":
    app.run(port=5000)
