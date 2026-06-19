from ultralytics import YOLO
import glob

new_names = {0: '작물', 1: '잡초'}

model_paths = glob.glob('runs/**/weights/*.pt', recursive=True)

for path in model_paths:
    model = YOLO(path)
    print(f"Before: {model.names}")
    model.model.names = new_names
    model.save(path)
    print(f"After:  {YOLO(path).names}")
    print(f"Saved: {path}\n")

print("완료!")
