# -*- coding: utf-8 -*-
from ultralytics import YOLO
import json

model= YOLO("models/best_mixed_train.pt")
results= model("samples/test.jpg")

r= results[0]
print(r.names)
print(r.boxes.xyxy)
print(r.boxes.conf)
print(r.boxes.cls)

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

print(json.dumps(results_to_json(r), indent=2))