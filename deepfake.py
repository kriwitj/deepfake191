import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# ตั้งค่าการใช้งานเฉพาะ CPU
# providers = ['CPUExecutionProvider']
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']


# โหลด face detector
print("🔍 Loading face detector...")
face_app = FaceAnalysis(name='buffalo_l', providers=providers)
# face_app.prepare(ctx_id=0)
face_app.prepare(ctx_id=0, det_size=(640, 640))

# โหลด face swapper
swapper_model_path = os.path.join("models", "inswapper_128.onnx")
if not os.path.exists(swapper_model_path):
    raise FileNotFoundError(f"Swapper model not found: {swapper_model_path}")

print("🔁 Loading face swapper...")
swapper = get_model(swapper_model_path, providers=providers)

# โหลดภาพ source face และตรวจจับ
# source_path = "images/gongyu.jpg"
# source_path = "images/Cha_Eun-woo.png"
# source_path = "images/Cha_Eun-woo.jpg"
# source_path = "images/nunew.jpeg"
source_path = "images/prayut.jpg"
# source_path = "images/tono.jpg"
if not os.path.exists(source_path):
    raise FileNotFoundError(f"Source image not found: {source_path}")

source_img = cv2.imread(source_path)
source_faces = face_app.get(source_img)
if not source_faces:
    raise RuntimeError("No face detected in source image.")
source_face = source_faces[0]

# เปิดกล้อง
print("🎥 Starting webcam...")
cap = cv2.VideoCapture(0)  # ใช้ 0 สำหรับกล้องหลัก
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Background
# bg_img = cv2.GaussianBlur(frame, (55, 55), 0)

bg_path = "images/tono.jpg"
bg_img = cv2.imread(bg_path)
bg_img = cv2.resize(bg_img, (640, 480))

if not cap.isOpened():
    raise RuntimeError("❌ Cannot open webcam.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ตรวจจับใบหน้าในเฟรม
    faces = face_app.get(frame)
    for face in faces:
        # สลับใบหน้า
        try:
            frame = swapper.get(frame, face, source_face, paste_back=True)
        except Exception as e:
            print(f"⚠️ Error in swapping: {e}")

    # แสดงผล
    cv2.imshow("Realtime Deepfake Face Swap", frame)

    # กด 'q' เพื่อออก
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้องและหน้าต่าง
cap.release()
cv2.destroyAllWindows()
print("✅ Webcam stopped.")
