from flask import Flask, request

app = Flask(__name__)


@app.route("/medicines/identify", methods=["POST"])
def identifyMedicines():
    image_file = request.files["file"]

    """
    1. image_file을 모델의 입력 형식에 부합하는 데이터로 변환
    2. 모델 추론
    3. 식별 결과 리턴
    """

    image_file.save(f"./upload/{image_file.filename}")

    return "success"


if __name__ == "__main__":
    app.run(port=5000)
