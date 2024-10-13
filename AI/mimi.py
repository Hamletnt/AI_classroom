from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/submit', methods=['POST'])
def submit():
    data = request.get_json()  # รับข้อมูล JSON จาก request
    print(data)  # แสดงข้อมูลใน console ของ Flask
    # ทำการประมวลผลข้อมูลที่ส่งมา (เช่น บันทึกลงฐานข้อมูล)
    
    return jsonify({'message': 'Data received successfully!'}), 200

if __name__ == '__main__':
    app.run(debug=True)
