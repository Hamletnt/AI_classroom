let currentIndex = 0; // เก็บตำแหน่งของรูปภาพปัจจุบัน
let images = []; // เก็บไฟล์รูปภาพที่อัปโหลด

document.addEventListener('DOMContentLoaded', function() {
    const uploadedImages = JSON.parse(localStorage.getItem('uploadedImages')) || [];

    if (uploadedImages.length > 0) {
        images = uploadedImages; // ใช้ภาพจาก local storage

        // แสดงภาพที่อัปโหลดแรก
        displayImage(0); // เรียกใช้ displayImage โดยไม่รอ

        // ตั้งค่าฟังก์ชันสำหรับปุ่มถัดไป
        document.getElementById('nextButton').addEventListener('click', function() {
            currentIndex = (currentIndex + 1) % images.length; 
            displayImage(currentIndex);
        });

        // ตั้งค่าฟังก์ชันสำหรับปุ่มย้อนกลับ
        document.getElementById('prevButton').addEventListener('click', function() {
            currentIndex = (currentIndex - 1 + images.length) % images.length; 
            displayImage(currentIndex);
        });
    } else {
        // หากไม่มีภาพ ให้แสดงข้อความหรือจัดการกับ UI ตามที่คุณต้องการ
        console.log("ไม่มีภาพที่อัปโหลด");
    }
});

// ฟังก์ชันสำหรับแสดงรูปภาพ
function displayImage(index) {
    const currentImage = document.getElementById('currentImage');
    currentImage.src = images[index]; // เปลี่ยน src ของรูปที่แสดง
}

// ลบ localStorage เมื่อปิดหน้าเว็บ
window.addEventListener('beforeunload', function() {
    // รีเซ็ตค่า currentIndex หรือ ลบข้อมูลที่เกี่ยวข้อง
    localStorage.removeItem('uploadedImages'); // ลบข้อมูลรูปภาพใน localStorage
    currentIndex = 0; // รีเซ็ตค่า currentIndex
});
