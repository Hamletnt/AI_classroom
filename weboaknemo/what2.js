let currentIndex = 0; // เก็บตำแหน่งของรูปภาพปัจจุบัน
let images = []; // เก็บไฟล์รูปภาพที่อัปโหลด

document.addEventListener('DOMContentLoaded', function() {
    // ดึง URL ของภาพจาก localStorage
    const uploadedImage = localStorage.getItem('uploadedImage');
    const currentImage = document.getElementById('currentImage');
    
    // แสดงผลรูปภาพที่อัปโหลด
    if (uploadedImage) {
        currentImage.src = uploadedImage;
    } else {
        currentImage.alt = "ไม่พบภาพที่อัปโหลด";
    }

    const predictionResult = JSON.parse(localStorage.getItem('predictionResult'));
    const categoryLabel = document.getElementById('categoryLabel');
    const confidenceLabel = document.getElementById('confidenceLabel');

    if (predictionResult && !isNaN(predictionResult.Confidence)) {
        categoryLabel.innerText = `Family : ${predictionResult.Predicted}`;
        confidenceLabel.innerText = `Confidence : ${predictionResult.Confidence.toFixed(2)}%`;
    } else {
        categoryLabel.innerText = 'Family : ไม่พบผลการพยากรณ์';
        confidenceLabel.innerText = 'Confidence : ไม่พบผลการพยากรณ์';
    }
});


// ฟังก์ชันสำหรับแสดงรูปภาพ
function displayImage(index) {
    const currentImage = document.getElementById('currentImage');
    currentImage.src = images[index]; // เปลี่ยน src ของรูปที่แสดง
    
    // แสดง URL ของรูปภาพใน console
    console.log("กำลังแสดงรูปภาพ:", images[index]);

    // สร้างองค์ประกอบ img เพื่อแสดงใน console (อาจไม่แสดงในบางเบราว์เซอร์)
    console.log('%c ', `font-size: 100px; background: url(${images[index]}) no-repeat; background-size: contain;`);
}

// ลบ localStorage เมื่อปิดหน้าเว็บ
window.addEventListener('beforeunload', function() {
    // รีเซ็ตค่า currentIndex หรือ ลบข้อมูลที่เกี่ยวข้อง
    localStorage.removeItem('uploadedImages'); // ลบข้อมูลรูปภาพใน localStorage
    currentIndex = 0; // รีเซ็ตค่า currentIndex
});
