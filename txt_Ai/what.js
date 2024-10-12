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
                images.push(e.target.result); // เก็บ URL ของรูปที่อ่านแล้วใน array

                // บันทึกภาพลงใน local storage
                localStorage.setItem('uploadedImages', JSON.stringify(images));

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
