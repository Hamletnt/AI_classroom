let currentIndex = 0; // เก็บตำแหน่งของรูปภาพปัจจุบัน
let images = []; // เก็บไฟล์รูปภาพที่อัปโหลด

// เมื่อผู้ใช้คลิกที่ปุ่ม Upload image
document.getElementById('uploadButton').addEventListener('click', function() {
    document.getElementById('fileInput').click(); // เปิดกล่องเลือกไฟล์
});

// เมื่อมีการเลือกไฟล์
document.getElementById('fileInput').addEventListener('change', function(event) {
    const files = event.target.files;

    if (files.length > 0) {
        images = []; // ล้างข้อมูลรูปเก่า
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const reader = new FileReader();
            reader.onload = function(e) {
                const imageDataUrl = e.target.result;
                images.push(imageDataUrl); // เก็บ URL ของรูปที่อ่านแล้วใน array

                // เก็บ URL ของภาพลงใน localStorage
                localStorage.setItem('uploadedImage', imageDataUrl);

                if (images.length === 1) {
                    displayImage(0); // ถ้าอัปโหลดรูปแรกแสดงรูปแรกทันที
                }
            };
            reader.readAsDataURL(file);
        }
    }
});

// ฟังก์ชันสำหรับแสดงรูปภาพ
function displayImage(index) {
    const currentImage = document.getElementById('currentImage');
    currentImage.src = images[index]; // เปลี่ยน src ของรูปที่แสดง
}

// เมื่อผู้ใช้กดปุ่มเลื่อนไปข้างหน้า
document.getElementById('nextButton').addEventListener('click', function() {
    if (images.length > 0) {
        currentIndex = (currentIndex + 1) % images.length; // เพิ่ม index และวนกลับไปที่รูปแรกถ้าถึงรูปสุดท้าย
        displayImage(currentIndex);
    }
});

// เมื่อผู้ใช้กดปุ่มเลื่อนกลับไปข้างหลัง
document.getElementById('prevButton').addEventListener('click', function() {
    if (images.length > 0) {
        currentIndex = (currentIndex - 1 + images.length) % images.length; // ลด index และวนกลับไปที่รูปสุดท้ายถ้าถอยไปก่อนรูปแรก
        displayImage(currentIndex);
    }
});

// อาร์เรย์สำหรับประเภทไฟล์ที่อนุญาต
const allowedExtensions = ['jpg', 'jpeg', 'png'];

// ฟังก์ชันสำหรับการส่งภาพไปยัง API
document.getElementById('searchButton').addEventListener('click', async function() {
    if (images.length === 0) {
        alert('กรุณาเลือกภาพก่อน');
        return;
    }

    const formData = new FormData();
    const currentImageData = images[currentIndex];
    const blob = await (await fetch(currentImageData)).blob(); // แปลง Data URL เป็น Blob

    // ตรวจสอบประเภทไฟล์จาก Blob
    const fileType = blob.type; // ดึงประเภทไฟล์จาก Blob
    const allowedMimeTypes = ['image/jpeg', 'image/png']; // กำหนดประเภท MIME ที่อนุญาต

    // ตรวจสอบประเภทไฟล์
    if (!allowedMimeTypes.includes(fileType)) {
        alert('ประเภทไฟล์ไม่ถูกต้อง');
        return;
    }

    const fileExtension = fileType.split('/').pop(); // ดึงนามสกุลจากประเภท MIME
    const fileName = `image.${fileExtension}`; // ตั้งชื่อไฟล์ตามนามสกุลที่ถูกต้อง
    formData.append('file', blob, fileName); // ส่งไฟล์ภาพไปยัง API

    try {
        const response = await fetch('http://3.107.26.166:8000/predict', {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();
        if (result.Predicted && result.Confidence) {
            // บันทึกข้อมูลลงใน localStorage
            const confidenceValue = parseFloat(result.Confidence.replace('%', ''));
            localStorage.setItem('predictionResult', JSON.stringify({
                Predicted: result.Predicted,
                Confidence: confidenceValue
            }));

            window.location.href = 'what2.html'; // เปลี่ยนไปยังหน้า what2.html
        } else {
            alert(`เกิดข้อผิดพลาด: ${result.error}`);
        }
    } catch (error) {
        console.error('เกิดข้อผิดพลาดในการพยากรณ์:', error);
        alert('การพยากรณ์ล้มเหลว');
    }
});