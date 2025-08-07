# Ada Lovelace School Assistant

This project is a web application that provides an AI-powered assistant for the Ada Lovelace School. The assistant can answer questions about the school based on the information provided to it.

## About the Project

The project consists of two main components:

*   **A Flask API:** This API is responsible for processing and embedding documents. It provides an endpoint to upload documents, which are then processed, embedded, and stored in a MongoDB database.
*   **A Streamlit web app:** This web app provides a user interface for the AI assistant. Users can log in and ask questions to the assistant. The assistant uses the embedded documents to find relevant information and generate a response.

## Getting Started

To get started with the project, you will need to have the following installed:

*   Python 3.10+
*   pip
*   Docker (optional)

### Installation

1.  Clone the repository:
    ```
    git clone https://github.com/example/ada-lovelace-school-assistant.git
    ```
2.  Install the dependencies:
    ```
    pip install -r requirements.txt
    ```
3.  Set up the environment variables:
    Create a `.env` file in the root of the project and add the following environment variables:
    ```
    MONGODB_URI=<your_mongodb_uri>
    HF_TOKEN=<your_huggingface_token>
    GEMINI_API_KEY=<your_gemini_api_key>
    PORT=8080
    API_TOKEN=<your_api_token>
    VALID_CREDENTIALS='{"admin": "Password1!", "teacher": "Password2!", "user": "Password3!"}'
    ```

## Usage

### Running the API

To run the API, use the following command:
```
python src/api/server.py
```

The API will be available at `http://localhost:8080`.

### Running the Web App

To run the web app, use the following command:
```
streamlit run src/web_app/app.py
```

The web app will be available at `http://localhost:8501`.

## Running the Tests

To run the tests, use the following command:
```
pytest
```
