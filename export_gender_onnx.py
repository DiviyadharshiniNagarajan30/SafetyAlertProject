import torch
import torch.nn as nn
from torchvision import models
import os

# -------------------------------
# Paths
# -------------------------------
pth_model_path = r"C:\Users\ndivi\SafetyAlertProject\models\gender_classifier_test.pth"
onnx_model_path = r"C:\Users\ndivi\SafetyAlertProject\models\gender_classifier_test.onnx"

# -------------------------------
# Load model
# -------------------------------
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # 2 classes: male, female

model.load_state_dict(torch.load(pth_model_path, map_location="cpu"))
model.eval()

# -------------------------------
# Dummy input for ONNX export
# -------------------------------
dummy_input = torch.randn(1, 3, 224, 224)  # batch size 1, 3 channels, 224x224 image

# -------------------------------
# Export to ONNX
# -------------------------------
os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)

torch.onnx.export(
    model,
    dummy_input,
    onnx_model_path,
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
)

print(f"ONNX model exported to: {onnx_model_path}")
