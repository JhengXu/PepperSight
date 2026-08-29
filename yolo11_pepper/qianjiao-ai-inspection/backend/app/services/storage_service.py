from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
from fastapi import HTTPException, UploadFile


PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
MAX_IMAGE_BYTES = 12 * 1024 * 1024


def _public_url(filename: str) -> str:
    return f"/uploads/{filename}"


async def save_upload_image(upload: UploadFile, prefix: str = "detection") -> str:
    suffix = ALLOWED_IMAGE_TYPES.get(upload.content_type or "")
    if suffix is None:
        raise HTTPException(415, "图片仅支持 JPEG、PNG 或 WebP 格式")
    content = await upload.read(MAX_IMAGE_BYTES + 1)
    if not content:
        raise HTTPException(400, "上传图片不能为空")
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "图片大小不能超过 12MB")
    filename = f"{prefix}-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}{suffix}"
    (UPLOAD_DIR / filename).write_bytes(content)
    return _public_url(filename)


def save_camera_frame(frame, prefix: str = "camera") -> str:
    filename = f"{prefix}-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}.jpg"
    target = UPLOAD_DIR / filename
    ok = cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("摄像头截图保存失败")
    return _public_url(filename)
