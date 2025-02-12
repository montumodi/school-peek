from flask import Flask, request
import os

app = Flask(__name__)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return {"error": "No file provided"}, 400
    
    file = request.files["file"]
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(file_path)
    
    return {"filename": file.filename, "filepath": file_path}

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=80)
