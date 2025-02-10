import os
import requests
from flask import Flask, render_template, request, jsonify
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Hugging Face API 정보
HF_TOKEN = ""  # 본인의 Hugging Face Token
API_URL = "https://api-inference.huggingface.co/models/djuHm/interior-design-regist-model"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# Flask 라우트
@app.route("/")
def home():
    return render_template("index.html")  # 기본 웹페이지

@app.route("/generate", methods=["POST"])
def generate_image():
    try:
        data = request.get_json()  # JSON 형식으로 받기
        prompt = data.get("prompt")

        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        # Hugging Face API 요청
        payload = {"inputs": prompt}
        response = requests.post(API_URL, headers=headers, json=payload)

        # API 호출 상태 코드 및 메시지 확인
        print(f"Status Code: {response.status_code}")
        print(f"Response Text: {response.text}")

        if response.status_code == 200:
            image_data = response.content
            image = Image.open(BytesIO(image_data))

            # 생성된 이미지를 저장하고 경로 반환
            output_path = os.path.join("static", "generated_image.png")
            image.save(output_path)

            return jsonify({"image_url": f"/static/generated_image.png"})
        else:
            return jsonify({"error": f"API request failed with status {response.status_code}, {response.text}"}), 500

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


# CORS 설정 (필요한 경우에만 사용)
from flask_cors import CORS
CORS(app)

# Flask 앱 실행
if __name__ == "__main__":
    app.run(debug=True)
