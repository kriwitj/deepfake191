
# 🧑‍🔬 Realtime Deepfake Face Swap (with InsightFace)

โปรเจกต์นี้เป็นการสลับใบหน้าบุคคลแบบเรียลไทม์ผ่านกล้องเว็บแคม โดยใช้โมเดลจาก InsightFace ซึ่งมีความแม่นยำสูงและประสิทธิภาพดีเยี่ยม

## ✅ คุณสมบัติ

- ตรวจจับใบหน้าแบบเรียลไทม์ด้วย `buffalo_l` face detection model
- สลับใบหน้าเป้าหมายจากภาพนิ่ง ไปยังใบหน้าในกล้องแบบ live
- รองรับ CPU และ GPU (CUDA) ด้วย ONNX Runtime
- ใช้งานง่าย ไม่ต้องเขียนโมเดลเอง

## 📁 โครงสร้างไฟล์

```
.
├── main.py                  # โค้ดหลัก
├── models/
│   └── inswapper_128.onnx   # โมเดล face swapper (ต้องดาวน์โหลดเอง)
├── images/
│   ├── prayut.jpg           # ภาพใบหน้าต้นทาง
│   └── tono.jpg             # ภาพพื้นหลังหรือภาพอื่นๆ
└── README.md
```

## ⚙️ วิธีติดตั้ง

### 1. Clone โปรเจกต์
```bash
git clone https://github.com/yourname/realtime-face-swap.git
cd realtime-face-swap
```

### 2. สร้าง Virtual Environment (แนะนำ)
```bash
python -m venv venv
source venv/bin/activate  # สำหรับ Linux/Mac
venv\Scripts\activate     # สำหรับ Windows
```

### 3. ติดตั้ง Dependency
```bash
pip install -r requirements.txt
```

หรือ:

```bash
pip install insightface onnxruntime-gpu opencv-python numpy
```

### 4. ดาวน์โหลดโมเดล `inswapper_128.onnx`
ดาวน์โหลดจาก:
> https://github.com/deepinsight/insightface/tree/master/model_zoo

นำไฟล์ `.onnx` มาใส่ในโฟลเดอร์ `models/`

## 🚀 วิธีใช้งาน

รันคำสั่ง:

```bash
python main.py
```

จากนั้นจะเปิดกล้อง พร้อมตรวจจับและสลับใบหน้าในเวลาจริง

### การกดออก
กดปุ่ม **`q`** บนคีย์บอร์ดเพื่อออกจากโปรแกรม

## 🧠 อธิบายโค้ดเบื้องต้น

- `FaceAnalysis(name='buffalo_l')` ใช้สำหรับตรวจจับใบหน้าด้วยโมเดลสำเร็จรูป
- `get_model('inswapper_128.onnx')` ใช้โมเดลสลับใบหน้าแบบ ONNX
- `face_app.get()` ตรวจจับใบหน้าในภาพหรือวิดีโอ
- `swapper.get(...)` ทำการสลับใบหน้าระหว่าง source กับ target

## 📌 หมายเหตุ

- หากไม่มี GPU ให้ใช้ `providers = ['CPUExecutionProvider']`
- ความละเอียดวิดีโอถูกตั้งไว้ที่ 640x480 เพื่อความเร็ว
- โมเดลจะถูกโหลดอัตโนมัติโดย InsightFace หากไม่พบในแคช จะดาวน์โหลดเอง

## 📜 License

MIT License
