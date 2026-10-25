# -*- coding: utf-8 -*-
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

app = FastAPI()
model = YOLO("models/best_mixed_train.pt")


def results_to_json(r):
    detections = []
    for box, conf, cls in zip(r.boxes.xyxy.tolist(),
                              r.boxes.conf.tolist(),
                              r.boxes.cls.tolist()):
        x1, y1, x2, y2 = box
        detections.append({
            "label": r.names[int(cls)],
            "confidence": round(conf, 4),
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "height": y2-y1,
            "width": x2-x1
        })
    return {"detections": detections}

@app.get("/")
def read_root():
    return {"message": "Debris detection API is running"}


@app.post("/detect")
async def detect(file: UploadFile = File()):
    contents = await file.read()
    try:
        image =  Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="The file entered is not an image.")   # write a clear message
    results = model(image)  
    return results_to_json(results[0])