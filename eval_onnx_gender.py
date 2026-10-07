import os, numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

model_path = r"C:\Users\ndivi\SafetyAlertProject\models\gender_classifier_test.onnx"
val_dir = r"C:\Users\ndivi\SafetyAlertProject\data\gender\val"

session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

classes = ["female","male"]

y_true = []
y_pred = []

for i, cls in enumerate(classes):
    cls_path = os.path.join(val_dir, cls)
    if not os.path.isdir(cls_path):
        continue
    for f in os.listdir(cls_path):
        if not f.lower().endswith((".jpg",".png",".jpeg")):
            continue
        img = Image.open(os.path.join(cls_path, f)).convert("RGB")
        x = transform(img).unsqueeze(0).numpy().astype(np.float32)
        outputs = session.run(None, {input_name: x})
        pred = int(np.argmax(outputs[0]))
        y_true.append(i)
        y_pred.append(pred)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\n=== ACCURACY ===")
total_acc = (y_true == y_pred).mean()
print("Overall accuracy:", total_acc)

female_acc = (y_pred[y_true == 0] == 0).mean() if np.sum(y_true==0)>0 else 0
male_acc   = (y_pred[y_true == 1] == 1).mean() if np.sum(y_true==1)>0 else 0

print("Female accuracy:", female_acc)
print("Male accuracy:", male_acc)

from sklearn.metrics import confusion_matrix
print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_true, y_pred))
