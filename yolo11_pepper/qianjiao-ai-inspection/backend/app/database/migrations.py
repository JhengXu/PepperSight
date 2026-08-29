from sqlalchemy import inspect

from app.database.db import engine
from app.models.entities import Detection


def migrate_detection_primary_key() -> bool:
    """Upgrade the pre-MJPEG demo schema without discarding saved detections."""
    inspector = inspect(engine)
    if "detections" not in inspector.get_table_names():
        return False
    columns = {column["name"]: column for column in inspector.get_columns("detections")}
    id_type = type(columns["id"]["type"]).__name__.upper()
    if "sample_code" in columns and id_type == "INTEGER":
        return False

    legacy_columns = [
        "batch_id",
        "timestamp",
        "image_url",
        "annotated_image_url",
        "variety",
        "length",
        "width",
        "color_score",
        "integrity_score",
        "shape_score",
        "size_score",
        "defect_score",
        "quality_score",
        "grade",
        "confidence",
        "defects",
        "processing_time",
        "grade_reason",
    ]
    with engine.begin() as connection:
        table_names = {
            row[0]
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "detections_legacy" in table_names:
            raise RuntimeError("检测到未完成的数据库迁移，请先备份 data/qianjiao.db")

        connection.exec_driver_sql("ALTER TABLE detections RENAME TO detections_legacy")
        legacy_indexes = connection.exec_driver_sql(
            "SELECT name FROM sqlite_master "
            "WHERE type='index' AND tbl_name='detections_legacy' AND sql IS NOT NULL"
        ).all()
        for (index_name,) in legacy_indexes:
            quoted_name = index_name.replace('"', '""')
            connection.exec_driver_sql(f'DROP INDEX "{quoted_name}"')

        Detection.__table__.create(bind=connection)
        target_columns = ", ".join(["sample_code", *legacy_columns])
        source_columns = ", ".join(["id", *legacy_columns])
        connection.exec_driver_sql(
            f"INSERT INTO detections ({target_columns}) "
            f"SELECT {source_columns} FROM detections_legacy"
        )
        connection.exec_driver_sql("DROP TABLE detections_legacy")
    return True


def migrate_detection_source() -> bool:
    """Tag pre-integration rows as legacy without deleting user data."""
    inspector = inspect(engine)
    if "detections" not in inspector.get_table_names():
        return False
    columns = {column["name"] for column in inspector.get_columns("detections")}
    if "source" in columns:
        return False
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "ALTER TABLE detections ADD COLUMN source VARCHAR(24) NOT NULL DEFAULT 'legacy'"
        )
        connection.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS ix_detections_source ON detections (source)"
        )
    return True
