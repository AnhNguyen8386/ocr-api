import torch
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from collections import defaultdict
import os

SEGMENT_MODEL_PATH = "app/models/model_segment.pt"
CLASSIFY_MODEL_PATH = "app/models/model_classify.pth"
DETECT_MODEL_PATH = "app/models/detect_model.pt"

def load_vietocr():
    config = Cfg.load_config_from_name("vgg_transformer")
    config['device'] = 'cpu'
    config['predictor']['beamsearch'] = True
    return Predictor(config)

vietocr = load_vietocr()
yolo_segment = YOLO(SEGMENT_MODEL_PATH)
yolo_detect = YOLO(DETECT_MODEL_PATH)

model_rotate = models.resnet18(weights=None)
model_rotate.fc = torch.nn.Linear(model_rotate.fc.in_features, 2)
model_rotate.load_state_dict(torch.load(CLASSIFY_MODEL_PATH, map_location="cpu"))
model_rotate.eval()

def detect_cmqs_region(image):
    results = yolo_segment.predict(image, conf=0.5)
    if not results or not results[0].boxes:
        return None
    x1, y1, x2, y2 = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
    return (x1, y1, x2, y2)

def crop_image(image, box):
    return image.crop(box)

def fix_rotation(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        pred = torch.argmax(model_rotate(input_tensor), dim=1).item()
    return image.rotate(180) if pred == 1 else image

def detect_fields(image):
    results = yolo_detect.predict(image, conf=0.4)[0]
    fields = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = results.names[int(box.cls[0])]
        fields.append({'label': label, 'box': (x1, y1, x2, y2)})
    return fields

def ocr_boxes(image, fields):
    for field in fields:
        x1, y1, x2, y2 = field['box']
        crop = image.crop((x1, y1, x2, y2))
        text = vietocr.predict(crop)
        field['text'] = text
        field['cx'] = (x1 + x2) / 2
        field['cy'] = (y1 + y2) / 2
    return fields

def merge_boxes_by_label_and_position(fields, y_thresh=25):
    grouped = defaultdict(list)
    for field in fields:
        grouped[field['label']].append(field)

    merged_result = {}
    for label, items in grouped.items():

        items.sort(key=lambda f: f['cy'])

        lines = []
        current_line = []

        for field in items:
            if not current_line:
                current_line.append(field)
                continue

            last = current_line[-1]
            same_line = abs(field['cy'] - last['cy']) < y_thresh

            if same_line:
                current_line.append(field)
            else:
                current_line.sort(key=lambda f: f['cx'])
                lines.append(' '.join(w['text'] for w in current_line))
                current_line = [field]

        if current_line:
            current_line.sort(key=lambda f: f['cx'])
            lines.append(' '.join(w['text'] for w in current_line))

        merged_result[label] = ' '.join(lines)

    return merged_result

def process_image(image: Image.Image):
    region_box = detect_cmqs_region(image)
    if region_box is None:
        raise ValueError("Không phát hiện được vùng CMQS")

    image = crop_image(image, region_box)
    image = fix_rotation(image)
    fields = detect_fields(image)
    fields = ocr_boxes(image, fields)
    result = merge_boxes_by_label_and_position(fields)

    label_order = ['Số ID', 'Họ tên', 'Cấp bậc', 'Đơn vị cấp', 'Hạn sử dụng']
    ordered_result = {label: result[label] for label in label_order if label in result}
    return ordered_result
