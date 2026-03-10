from PIL import Image, ExifTags
from datetime import datetime

def get_exif_dict(path: str):
    try:
        img = Image.open(path)
        exif = img._getexif() or {}
        return {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
    except Exception:
        return {}

def parse_datetime_original(exif: dict):
    try:
        dt_str = exif.get("DateTimeOriginal") or exif.get("DateTime")
        if dt_str:
            return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S").strftime("%Y-%m-%d")
    except Exception:
        return None
    return None

def camera_make_model(exif: dict):
    return exif.get("Make"), exif.get("Model")
