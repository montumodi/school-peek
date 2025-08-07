import pytest
from flask import Flask
import sys
import os
from unittest.mock import patch

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def app():
    # Mock environment variables before importing the app
    with patch.dict(os.environ, {
        "MONGODB_URI": "mongodb://localhost:27017/",
        "HF_TOKEN": "test_hf_token",
        "GEMINI_API_KEY": "test_gemini_api_key",
        "PORT": "8080",
        "API_TOKEN": "test_api_token",
        "VALID_CREDENTIALS": "{}"
    }):
        from src.api.routes.upload import upload_blueprint
        app = Flask(__name__)
        app.register_blueprint(upload_blueprint)
        yield app

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def client(app):
    return app.test_client()

def test_upload_no_token(client):
    response = client.post('/upload')
    assert response.status_code == 401
    assert response.json == {"error": "Unauthorized"}

def test_upload_with_token(client):
    with patch('src.api.routes.upload.process_text_content') as mock_process_text_content:
        mock_process_text_content.return_value = ({"message": "ok"}, 200)
        response = client.post('/upload', headers={'Authorization': 'Bearer test_api_token'}, data={'email_body': 'test'})
        assert response.status_code == 200

import io

def test_upload_file_with_token(client):
    with patch('src.api.routes.upload.process_file') as mock_process_file:
        mock_process_file.return_value = ({"message": "ok"}, 200)
        data = {
            'file': (io.BytesIO(b'my-file-contents'), 'test_file.txt')
        }
        response = client.post(
            '/upload',
            headers={'Authorization': 'Bearer test_api_token'},
            content_type='multipart/form-data',
            data=data
        )
        assert response.status_code == 200
        assert response.json['summary']['processed_files'] == 1
