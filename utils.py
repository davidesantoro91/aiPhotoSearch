import os
from typing import List, Tuple
from PIL import Image

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def list_images(root: str) -> Tuple[list, list]:
    paths = []
    names = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in VALID_EXTS:
                p = os.path.join(dirpath, fn)
                paths.append(p)
                names.append(os.path.relpath(p, root))
    return paths, names

def load_image(path: str, max_side: int = 768) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max(w, h) / max_side if max(w, h) > max_side else 1.0
    if scale > 1.0:
        img = img.resize((int(w/scale), int(h/scale)))
    return img

def pil_from_bytes(data: bytes) -> Image.Image:
    from io import BytesIO
    return Image.open(BytesIO(data)).convert("RGB")
