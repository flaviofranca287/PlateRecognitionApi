from flask import Flask, render_template, Response
import cv2
import os
import pytesseract
import re
from collections import Counter
import requests


os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'
API_URL = "http://localhost:5001/payments"

amount = 20
documentNumber = "12345678901"

app = Flask(__name__)

plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def capture_plates():
    cam = cv2.VideoCapture(0)  # 0 para webcam padrão
    last_text = None
    
    if cam.isOpened():
        print("Conectou")
        validation, frame = cam.read()
        while validation:
            validation, frame = cam.read()
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in plates:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                plate_img = frame[y:y+h, x:x+w]
                text = perform_ocr_with_voting(plate_img)
                
                if text and (last_text is None or is_significantly_different(text, last_text)):
                    cv2.imwrite("PlacaDetectada.png", plate_img)
                    print(f"Placa detectada: {text}")
                    
                    send_plate_to_api(text)  # Envia a placa para a API

                    last_text = text
            
            cv2.imshow("Video da Webcam", frame)
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break

    cam.release()
    cv2.destroyAllWindows()

def preprocess_image(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    resized = cv2.resize(binary, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    return blurred

def recognize_text(plate_img):
    preprocessed = preprocess_image(plate_img)
    custom_config = r'--oem 3 --psm 11'
    text = pytesseract.image_to_string(preprocessed, config=custom_config)
    text = re.sub(r'[^A-Za-z0-9]', '', text)  # Remove caracteres indesejados
    return text.strip()

def perform_ocr_with_voting(plate_img):
    readings = []
    
    for _ in range(3):  # Realiza 10 leituras
        text = recognize_text(plate_img)
        if is_valid_reading(text):
            readings.append(text)
    
    if readings:
        most_common_text, _ = Counter(readings).most_common(1)[0]  # Obtém o texto mais frequente
        return most_common_text
    return None

def is_valid_reading(text):
    # Define os critérios de validação para formatos ABC1234 e ABC1C34
    pattern = re.compile(r'^[A-Za-z]{3}[0-9]{4}$|^[A-Za-z]{3}[0-9][A-Za-z][0-9]{2}$')
    return bool(pattern.match(text))

def is_significantly_different(new_text, old_text):
    # Verifica se pelo menos metade dos caracteres é diferente
    difference_count = sum(1 for a, b in zip(new_text, old_text) if a != b)
    return difference_count >= len(new_text) / 2

def send_plate_to_api(plate_text):
    global amount, documentNumber
    # Dados que serão enviados na requisição
    data = {
        'plate': plate_text,
        'amount': amount,
        'documentNumber': documentNumber
    }
    try:
        response = requests.post(API_URL, json=data)
        if response.status_code == 201:
            print(f"Pagamento para a placa {plate_text} efetuado com sucesso!")
        else:
            print(f"Falha ao tentar realizar pagamento. Placa: {plate_text}. Código de status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Erro ao conectar com a API: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(capture_plates(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
