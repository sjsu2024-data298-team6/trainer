import torch
from ultralytics import YOLO
import json


def train_main():
    with open("params.json", "r") as f:
        params = json.loads(f.read())

    epochs = params["epochs"] if "epochs" in params.keys() else 10
    imgsz = params["imgsz"] if "imgsz" in params.keys() else 640

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO("yolo11n.pt")
    results = model.train(
        data="./data/data.yaml",
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        device=device,
    )
