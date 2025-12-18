import os
import shutil

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED_EXTENSIONS = {".txt", ".md"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def validate_file(file) -> tuple[bool, str]:
    """Validate file size and extension"""
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size > MAX_FILE_SIZE:
        return False, f"File too large. Max size is {MAX_FILE_SIZE / (1024*1024)}MB"

    return True, ""

def save_file(file):

    valid, error = validate_file(file)

    if not valid:
        raise Exception(error)

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Save the file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "message": f"File '{file.filename}' uploaded successfully",
        "path": file_path,
        "size": os.path.getsize(file_path)
    }, file_path