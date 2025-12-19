import os
import shutil
import re

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".txt",
    ".md",
    ".rst",
    ".csv",
    ".tsv",
    ".log",
    ".ini",
    ".cfg",
    ".conf",
    ".yaml",
    ".yml",
    ".json",
    ".xml",
    ".html",
    ".htm",
    ".css",
    ".js",
    ".ts",
    ".py",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".sh",
    ".bat",
    ".ps1",
    ".sql",
    ".tex",
    ".bib",
    ".properties",
    ".env",
    ".mjs"
}

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def sanitize_filename(filename: str) -> str:
    # Extract the base name only (removes path traversal)
    filename = os.path.basename(filename)

    # Normalize filename (allow only safe characters)
    filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)

    # Prevent empty filenames
    if not filename:
        raise ValueError("Invalid filename")

    return filename


def validate_file(file) -> tuple[bool, str]:
    """Validate file size and extension"""
    # Check file size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        return False, f"File too large. Max size is {MAX_FILE_SIZE / (1024*1024)}MB"

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, "Invalid file extension"

    return True, ""

def save_file(file):

    valid, error = validate_file(file)

    if not valid:
        raise Exception(error)

    file.filename = sanitize_filename(file.filename)
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