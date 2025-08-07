from flask import Flask
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from api.routes.upload import upload_blueprint
from config.config import PORT

app = Flask(__name__)

app.register_blueprint(upload_blueprint)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=PORT)
