from flask import Blueprint, request, jsonify
from functools import wraps
from src.config.config import API_TOKEN
from src.api.services.processing_service import process_text_content, process_file

upload_blueprint = Blueprint('upload', __name__)

def require_api_token(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.headers.get('Authorization') != f'Bearer {API_TOKEN}':
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function

@upload_blueprint.route("/upload", methods=["POST"])
@require_api_token
def upload_files():
    """Handle email content processing - both email body and attachments"""
    email_body = request.form.get("email_body")
    email_subject = request.form.get("email_subject") or "Email Content"

    print(f"email_body: {email_body}")

    files = []
    for key, file_obj in request.files.items():
        if file_obj and file_obj.filename:
            files.append(file_obj)

    print(f"files: {files}")

    if not email_body and not files:
        return jsonify({"error": "No email body or files provided"}), 400

    dry_run = request.args.get("dry_run", "false").lower() == "true"

    if dry_run:
        response = []
        if email_body:
            response.append({"type": "email_body", "subject": email_subject, "content_length": len(email_body)})
        if files:
            file_info = []
            for file in files:
                content = file.read()
                file.seek(0)
                file_info.append({
                    "type": "attachment",
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "size": len(content)
                })
            response.extend(file_info)
        return jsonify({"preview": response, "total_files": len(files)})

    response = []
    processed_files = 0
    failed_files = 0

    if email_body:
        try:
            result = process_text_content(email_body, email_subject, "email_body")
            response.append({"type": "email_body", "result": result})
        except Exception as e:
            response.append({"type": "email_body", "result": {"error": str(e)}, "status": "failed"})

    if files:
        for i, file in enumerate(files):
            try:
                result = process_file(file)
                response.append({"type": "attachment", "filename": file.filename, "result": result})
                if result[1] == 200:
                    processed_files += 1
                else:
                    failed_files += 1
            except Exception as e:
                response.append({"type": "attachment", "filename": file.filename, "result": {"error": str(e)}, "status": "failed"})
                failed_files += 1

    return jsonify({
        "results": response,
        "summary": {
            "total_files": len(files),
            "processed_files": processed_files,
            "failed_files": failed_files,
            "email_body_processed": bool(email_body)
        }
    })
