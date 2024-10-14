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

async function fetchDetailsFromGoogleSheets(predictedCategory) {
    const sheetId = '1WEH5IBw-FJRdMZRFtl695eLVu46SpvGtW4ZEw3mN_3c'; // Google Sheets ID
    const sheetRange = 'Sheet1'; // ชื่อชีต
    const apiKey = 'AIzaSyAEhw01BV1z5jIell5tAVVhOBLgsPlCTFw'; // API Key
    
    try {
        const response = await fetch(`https://sheets.googleapis.com/v4/spreadsheets/${sheetId}/values/${sheetRange}?key=${apiKey}`);
        const data = await response.json();
        console.log('Data from Google Sheets:', data); // ตรวจสอบข้อมูลที่ได้จาก Google Sheets
        
        const rows = data.values;

        // ค้นหาแถวที่ตรงกับประเภทที่พยากรณ์ไว้
        let selectedRow;
        for (const row of rows) {
            console.log('Row data:', row); // แสดงข้อมูลของแถวใน Google Sheets
            if (row[0] === predictedCategory) { // ค้นหาตรงกับหมวดหมู่ที่พยากรณ์ไว้
                selectedRow = row;
                break;
            }
        }

        console.log('Selected row:', selectedRow); // ตรวจสอบข้อมูลของแถวที่เลือก

        if (selectedRow) {
            // แสดงข้อมูลในหน้าเว็บ
            document.getElementById('details-frame').innerText = selectedRow[4] || 'ไม่มีข้อมูล'; // ดึงข้อมูลรายละเอียด
            document.getElementById('menu-frame').innerText = selectedRow[2] || 'ไม่มีข้อมูล'; // ดึงข้อมูลเมนู
            document.getElementById('drink-frame').innerText = selectedRow[3] || 'ไม่มีข้อมูล'; // ดึงข้อมูลเครื่องดื่ม
        } else {
            console.error('ไม่พบข้อมูลที่ตรงกับผลการพยากรณ์.');
        }
    } catch (error) {
        console.error('เกิดข้อผิดพลาดในการดึงข้อมูลจาก Google Sheets:', error);
    }
}


document.addEventListener('DOMContentLoaded', function() {
    const predictionResult = JSON.parse(localStorage.getItem('predictionResult'));
    
    if (predictionResult && !isNaN(predictionResult.Confidence)) {
        const predictedCategory = predictionResult.Predicted; // หมวดหมู่ที่พยากรณ์ได้

        // เรียกฟังก์ชันดึงข้อมูลจาก Google Sheets
        fetchDetailsFromGoogleSheets(predictedCategory);
    }
});

// ลบ localStorage เมื่อปิดหน้าเว็บ
window.addEventListener('beforeunload', function() {
    // รีเซ็ตค่า currentIndex หรือ ลบข้อมูลที่เกี่ยวข้อง
    localStorage.removeItem('uploadedImages'); // ลบข้อมูลรูปภาพใน localStorage
    currentIndex = 0; // รีเซ็ตค่า currentIndex
});