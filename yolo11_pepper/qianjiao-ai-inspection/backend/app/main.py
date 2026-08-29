import asyncio
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.router import router
from app.database.db import Base, SessionLocal, engine
from app.database.migrations import migrate_detection_primary_key, migrate_detection_source
from app.models.entities import GradingRule
from app.services.camera_pipeline import process_camera_frame
from app.services.camera_service import camera_service
from app.services.mock_service import mock_service
from app.services.model_service import model_service
from app.services.storage_service import UPLOAD_DIR


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    migrated = migrate_detection_primary_key()
    source_migrated = migrate_detection_source()
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        if db.get(GradingRule, 1) is None:
            db.add(GradingRule(id=1))
            db.commit()
    if migrated:
        logger.info("Detection 表已迁移为整数主键 + sample_code")
    if source_migrated:
        logger.info("Detection 表已增加来源字段；原有记录保留为 legacy")

    try:
        await asyncio.to_thread(model_service.load)
        logger.info("YOLO11 辣椒检测与分层分类模型加载完成")
    except Exception:
        logger.exception("YOLO11 模型加载失败；API 将继续启动并在健康检查中报告错误")

    loop = asyncio.get_running_loop()
    inference_lock = threading.Lock()
    inference_running = False

    def schedule_camera_detection(frame) -> None:
        nonlocal inference_running
        with inference_lock:
            if inference_running:
                return
            inference_running = True
        future = asyncio.run_coroutine_threadsafe(process_camera_frame(frame), loop)

        def report_failure(completed) -> None:
            nonlocal inference_running
            try:
                completed.result()
            except Exception:
                logger.exception("摄像头触发检测失败")
            finally:
                with inference_lock:
                    inference_running = False

        future.add_done_callback(report_failure)

    camera_service.start(schedule_camera_detection)
    try:
        yield
    finally:
        camera_service.stop()
        await mock_service.pause()


app = FastAPI(
    title="厉辣 API",
    description="贵州辣椒 AI 智能品质检测系统",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.include_router(router)


@app.get("/")
def root():
    return {"name": "厉辣", "docs": "/docs", "health": "/api/health"}
