from fastapi import FastAPI, File, UploadFile
from fastapi import Security, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
from tensorflow.keras.models import load_model
import numpy as np  
from PIL import Image
import io

# สร้าง instance ของ FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลดโมเดลของคุณ และโหลด class
model = load_model('./models/keras_model.h5') 
with open(r'./models/labels.txt', 'r') as f:
    class_labels = [line.strip() for line in f]

# ฟังก์ชันสำหรับการเตรียมข้อมูลจากรูปภาพก่อนส่งเข้าโมเดล
def preprocess_image(image: Image.Image):
    # แปลงภาพให้เป็น RGB เสมอ
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # ปรับขนาดของรูปภาพให้ตรงกับขนาด input ของโมเดล
    image = image.resize((224, 224))  # ตัวอย่าง, หากโมเดลต้องการขนาด 224x224
    image = np.array(image) / 255.0   # แปลงเป็น numpy array และปรับค่าความเข้มของสีให้อยู่ในช่วง 0-1
    image = np.expand_dims(image, axis=0)  # เพิ่มมิติใหม่เพื่อให้ตรงกับ input ของโมเดล (batch size)
    return image

# Route สำหรับการพยากรณ์จากรูปภาพที่ผู้ใช้ส่งมา
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # ตรวจสอบชนิดไฟล์
        if file.content_type not in ['image/jpeg', 'image/png']:
            return {"error": "Invalid file type."}

        # อ่านไฟล์รูปภาพ
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
    
        # เตรียมข้อมูลรูปภาพ
        processed_image = preprocess_image(image)
    
        # ส่งรูปภาพไปยังโมเดลเพื่อทำการพยากรณ์
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])  # หาประเภทที่มีค่าเป็นไปได้สูงสุด

        # ดึงค่าความมั่นใจ (confidence) ของคลาสที่โมเดลทำนายได้ดีที่สุด
        predicted_index = np.argmax(predictions[0])
        predicted_class = class_labels[predicted_index]
        confidence = predictions[0][predicted_index]
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence * 100:.2f}%")  
    
        # ส่งผลลัพธ์กลับไปในรูปแบบ JSON
        return {f"Predicted: {predicted_class} ({confidence * 100:.2f}%)"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)