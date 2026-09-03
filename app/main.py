"""
Shelly NAS Hub — self-hosted camera hub.

Features:
- Multi-camera support: RTSP (Reolink/IMOU/Hikvision/...), Shelly cameras (HTTP+WebRTC),
  Dahua NVR channels (CGI snapshot + RTSP).
- 24/7 continuous recording with FFmpeg (segmented MP4, configurable retention).
- AI object detection (YOLO26/YOLOv8) — person, car, truck, motorcycle, cat, dog.
- Face recognition with InsightFace (SCRFD detection + ArcFace 512-d embeddings).
- Face clustering, naming, and per-cluster crop management.
- Single-user authentication (bcrypt + signed cookies).
- Web UI: live grid, recordings timeline player, detection events, face clusters,
  camera CRUD, account management.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import httpx
import bcrypt
from itsdangerous import TimestampSigner, BadSignature, SignatureExpired
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, Response, Cookie, Form
from fastapi.responses import FileResponse, StreamingResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Конфигурация на logging — задължително за да се видят INFO logs от нашия app
# код в `docker logs` (по default uvicorn не override-ва logger level за app.* loggers,
# така че те остават на WARNING — губим face/recording/detection INFO logs).
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    force=True,
)
_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"


# ── настройки ──────────────────────────────────────────────────────────────
class NvrSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NVR_", case_sensitive=False)

    host:     str = ""        # 192.168.3.10
    user:     str = "admin"
    password: str = ""
    channels: str = "1,2,4,7"
    names:    str = "Камера 1,Камера 2,Камера 4,Камера 7"
    cache_s:  int = 10        # секунди кеш за snapshot


class DetectSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NVR_DETECT_", case_sensitive=False)

    enabled:           bool  = True
    model:             str   = "yolo26s.pt"  # yolo26n/s/m/l/x.pt — NVR_DETECT_MODEL (YOLO26 = +8% mAP, -32% CPU spri yolov8s)
    interval:          int   = 3      # секунди между проверки
    cooldown:          int   = 30     # минимум секунди между 2 event-а от 1 камера
    classes:           str   = "0,2,3,7,15,16"  # person,car,motorcycle,truck,cat,dog
    confidence:        float = 0.45   # default за повечето класове (car, dog, ...)
    confidence_person: float = 0.35   # по-нисък — хора рядко са false positive, за да не изпускаме
    confirm_frames:    int   = 2      # обектът трябва да е видян в N последователни кадъра преди да генерира event
    retention:         int   = 7      # дни
    hq:                bool  = True   # при детекция вземи HQ кадър (main stream) за запис


class RecordingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RECORDING_", case_sensitive=False)

    enabled:        bool = True
    path:           str  = "/recordings"
    retention_days: int  = 7
    segment_s:      int  = 60         # дължина на сегмент (sec) — 1 мин = удобно за seek/cleanup
    quality:        str  = "main"     # main | sub
    audio:          bool = True
    disk_limit_pct: int  = 90         # при > N% запълване → изтрий 1 ден retention


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False)
    data_dir: str = "/data"


class AuthSettings(BaseSettings):
    """Single-user authentication.

    enabled    : ако False — няма auth, всички requests са публични (default).
    user       : login username (default "admin").
    password_hash : bcrypt hash на паролата. Генериране:
                   `python3 -c "import bcrypt; print(bcrypt.hashpw(b'YOUR_PASS', bcrypt.gensalt()).decode())"`
    secret     : key за подписване на session cookies. Generate via
                 `python3 -c "import secrets; print(secrets.token_hex(32))"`.
                 Ако е празно — auto-generate при start (но при restart всички
                 sessions се invalidate-ват → препоръчва се да се зададе в .env).
    session_ttl_hours : валидност на session cookie (default 30 дни).
    """
    model_config = SettingsConfigDict(env_prefix="HOMEHUB_AUTH_", case_sensitive=False)

    enabled:           bool = False
    user:              str  = "admin"
    password_hash:     str  = ""
    secret:            str  = ""
    session_ttl_hours: int  = 24 * 30


auth_settings      = AuthSettings()
nvr_settings       = NvrSettings()
detect_settings    = DetectSettings()
recording_settings = RecordingSettings()
app_settings       = AppSettings()

DATA_DIR   = Path(app_settings.data_dir)
DB_PATH    = DATA_DIR / "history.db"
EVENTS_DIR = DATA_DIR / "events"
FACES_DIR  = DATA_DIR / "faces"

# COCO class id → BG етикет
_DETECT_LABELS: dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 7: "truck", 15: "cat", 16: "dog",
    17: "horse", 18: "sheep", 19: "cow", 24: "backpack",
}


# ── SQLite helpers (sync, извикват се в thread pool) ──────────────────────
def _db_init() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detection_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                channel     INTEGER NOT NULL,
                detected_at TEXT NOT NULL,
                labels      TEXT NOT NULL,
                confidences TEXT NOT NULL,
                image_path  TEXT
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_det_ch_ts "
            "ON detection_events(channel, detected_at)"
        )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detection_config (
                channel        INTEGER PRIMARY KEY,
                armed          INTEGER NOT NULL DEFAULT 1,
                min_confidence REAL    DEFAULT NULL
            )
        """)
        # Добавяме колоната ако таблицата вече съществува без нея (миграция)
        try:
            conn.execute("ALTER TABLE detection_config ADD COLUMN min_confidence REAL DEFAULT NULL")
        except Exception:
            pass
        # Bounding boxes за face recognition (миграция)
        try:
            conn.execute("ALTER TABLE detection_events ADD COLUMN boxes TEXT DEFAULT NULL")
        except Exception:
            pass
        # Разпознати лица (имена/cluster ids) — обогатява events ретроактивно
        try:
            conn.execute("ALTER TABLE detection_events ADD COLUMN faces TEXT DEFAULT NULL")
        except Exception:
            pass
        # Track ids — за обектен tracking (NEW/RETURN events)
        try:
            conn.execute("ALTER TABLE detection_events ADD COLUMN tracks TEXT DEFAULT NULL")
        except Exception:
            pass

        # ── Face recognition tables ────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS face_clusters (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT,
                created_at      TEXT NOT NULL,
                face_count      INTEGER DEFAULT 0,
                representative  TEXT,
                embedding       TEXT
            )
        """)
        # Миграция: ако таблицата вече съществува без embedding колоната
        try:
            conn.execute("ALTER TABLE face_clusters ADD COLUMN embedding TEXT")
        except Exception:
            pass
        # InsightFace ArcFace 512-d embeddings (replace dlib 128-d).
        # Колоните се добавят, но старите 128-d остават за fallback ако
        # InsightFace не се зареди.
        try:
            conn.execute("ALTER TABLE face_clusters ADD COLUMN embedding_v2 TEXT")
        except Exception:
            pass
        conn.execute("""
            CREATE TABLE IF NOT EXISTS face_crops (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                detected_at TEXT NOT NULL,
                channel     INTEGER NOT NULL,
                image_path  TEXT NOT NULL,
                embedding   TEXT NOT NULL,
                cluster_id  INTEGER REFERENCES face_clusters(id) ON DELETE SET NULL,
                event_id    INTEGER
            )
        """)
        try:
            conn.execute("ALTER TABLE face_crops ADD COLUMN embedding_v2 TEXT")
        except Exception:
            pass
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_face_crops_cluster "
            "ON face_crops(cluster_id)"
        )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        # ── Camera registry ─────────────────────────────────────────────────
        # Заменя hardcoded SHELLY_CAMS_CFG / EXTRA_CAMS_CFG / NVR env vars.
        # CRUD през /api/cameras endpoints. Channel е unique ID (стабилен
        # между restart-ове); recording paths и event paths използват channel.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cameras (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                channel        INTEGER UNIQUE NOT NULL,
                type           TEXT NOT NULL,         -- 'nvr_dahua' | 'rtsp_generic' | 'shelly'
                name           TEXT NOT NULL,
                enabled        INTEGER NOT NULL DEFAULT 1,
                has_audio      INTEGER NOT NULL DEFAULT 0,
                config         TEXT NOT NULL,         -- JSON със type-specific полета
                retention_days INTEGER,                -- per-camera override; NULL = use global
                created_at     TEXT NOT NULL,
                updated_at     TEXT NOT NULL
            )
        """)
        # Idempotent migration за съществуващи bases: ADD COLUMN IF NOT EXISTS
        # (SQLite не поддържа syntax-а, затова try/except)
        try:
            conn.execute("ALTER TABLE cameras ADD COLUMN retention_days INTEGER")
        except sqlite3.OperationalError:
            pass   # column already exists


# ── Camera registry helpers ─────────────────────────────────────────────────
# Cache: invalidate при insert/update/delete. Зарежда се lazy при първия
# достъп и при инициализация на background loops.
_cameras_cache: dict[int, dict] | None = None
_cameras_cache_lock = asyncio.Lock()


_CAM_SELECT_COLS = (
    "id, channel, type, name, enabled, has_audio, config, "
    "retention_days, created_at, updated_at"
)


def _camera_row_to_dict(row: tuple) -> dict:
    """Превръща SQL row в dict с parsed JSON config."""
    (cam_id, channel, ctype, name, enabled, has_audio, config_json,
     retention_days, created_at, updated_at) = row
    try:
        cfg = json.loads(config_json) if config_json else {}
    except json.JSONDecodeError:
        cfg = {}
    return {
        "id":             cam_id,
        "channel":        channel,
        "type":           ctype,
        "name":           name,
        "enabled":        bool(enabled),
        "has_audio":      bool(has_audio),
        "config":         cfg,
        "retention_days": retention_days,   # int | None (None = use global)
        "created_at":     created_at,
        "updated_at":     updated_at,
    }


def _db_list_cameras() -> list[dict]:
    """Връща списък от ВСИЧКИ камери (включително disabled), сортирани по channel."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            f"SELECT {_CAM_SELECT_COLS} FROM cameras ORDER BY channel"
        ).fetchall()
    return [_camera_row_to_dict(r) for r in rows]


def _db_get_camera_by_channel(channel: int) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            f"SELECT {_CAM_SELECT_COLS} FROM cameras WHERE channel = ?",
            (channel,),
        ).fetchone()
    return _camera_row_to_dict(row) if row else None


def _db_get_camera_by_id(cam_id: int) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            f"SELECT {_CAM_SELECT_COLS} FROM cameras WHERE id = ?",
            (cam_id,),
        ).fetchone()
    return _camera_row_to_dict(row) if row else None


def _db_insert_camera(
    channel: int, ctype: str, name: str,
    config: dict, has_audio: bool = False, enabled: bool = True,
    retention_days: int | None = None,
) -> int:
    """Insert + връща новия id. Хвърля sqlite3.IntegrityError при duplicate channel."""
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "INSERT INTO cameras(channel, type, name, enabled, has_audio, "
            "config, retention_days, created_at, updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (channel, ctype, name, 1 if enabled else 0,
             1 if has_audio else 0, json.dumps(config),
             retention_days, now, now),
        )
        return cur.lastrowid


def _db_update_camera(cam_id: int, fields: dict) -> bool:
    """Update только дадените полета. Връща True ако row е updated."""
    if not fields:
        return False
    sets: list[str] = []
    params: list = []
    for key in ("channel", "type", "name", "enabled", "has_audio", "retention_days"):
        if key in fields:
            sets.append(f"{key} = ?")
            v = fields[key]
            if key in ("enabled", "has_audio"):
                v = 1 if v else 0
            params.append(v)
    if "config" in fields:
        sets.append("config = ?")
        params.append(json.dumps(fields["config"]))
    if not sets:
        return False
    sets.append("updated_at = ?")
    params.append(datetime.now(timezone.utc).isoformat())
    params.append(cam_id)
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            f"UPDATE cameras SET {', '.join(sets)} WHERE id = ?",
            params,
        )
        return cur.rowcount > 0


def _db_delete_camera(cam_id: int) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("DELETE FROM cameras WHERE id = ?", (cam_id,))
        return cur.rowcount > 0


def _db_seed_cameras_if_empty() -> None:
    """При първо стартиране: ако таблицата е празна, импортира от env vars
    + hardcoded списъци за continuity. Идемпотентна — ако вече има камери,
    нищо не прави."""
    with sqlite3.connect(DB_PATH) as conn:
        count = conn.execute("SELECT COUNT(*) FROM cameras").fetchone()[0]
    if count > 0:
        return

    _LOGGER.info("Camera registry: seeding from env + hardcoded configs (first start)")

    # NVR Dahua канали — споделят host/user/password в nvr_settings
    if nvr_settings.host and nvr_settings.channels:
        nvr_chs   = [int(c.strip()) for c in nvr_settings.channels.split(",") if c.strip()]
        nvr_names = [n.strip() for n in nvr_settings.names.split(",")]
        for i, ch in enumerate(nvr_chs):
            name = nvr_names[i] if i < len(nvr_names) else f"Camera {ch}"
            cfg = {
                "host":       nvr_settings.host,
                "user":       nvr_settings.user,
                "password":   nvr_settings.password,
                "channel_id": ch,
            }
            try:
                _db_insert_camera(ch, "nvr_dahua", name, cfg, has_audio=False)
                _LOGGER.info("Seeded NVR camera ch=%d name=%s", ch, name)
            except sqlite3.IntegrityError:
                pass

    # Extra IP камери (Reolink/IMOU/др.) — pass-through от EXTRA_CAMS_CFG_DEFAULT
    for cam in EXTRA_CAMS_CFG_DEFAULT:
        cfg = {
            "rtsp_main": cam.get("rtsp_main", ""),
            "rtsp_sub":  cam.get("rtsp_sub", ""),
            "snap_url":  cam.get("snap_url", ""),
            "ip":        cam.get("ip", ""),
            "user":      cam.get("user", ""),
            "password":  cam.get("password", ""),
        }
        try:
            _db_insert_camera(
                cam["ch"], "rtsp_generic", cam["name"], cfg,
                has_audio=cam.get("has_audio", False),
            )
            _LOGGER.info("Seeded RTSP camera ch=%d name=%s", cam["ch"], cam["name"])
        except sqlite3.IntegrityError:
            pass

    # Shelly камери
    for cam in SHELLY_CAMS_CFG_DEFAULT:
        cfg = {"ip": cam["ip"]}
        try:
            _db_insert_camera(
                cam["ch"], "shelly", cam["name"], cfg, has_audio=False,
            )
            _LOGGER.info("Seeded Shelly camera ch=%d name=%s", cam["ch"], cam["name"])
        except sqlite3.IntegrityError:
            pass


def _cameras_get_all() -> dict[int, dict]:
    """Sync helper за зарежда + cache. {channel: camera_dict}."""
    global _cameras_cache
    if _cameras_cache is None:
        rows = _db_list_cameras()
        _cameras_cache = {c["channel"]: c for c in rows}
    return _cameras_cache


async def _cameras_refresh() -> dict[int, dict]:
    """Async refresh — извиква се след CRUD."""
    global _cameras_cache
    async with _cameras_cache_lock:
        rows = await asyncio.to_thread(_db_list_cameras)
        _cameras_cache = {c["channel"]: c for c in rows}
    return _cameras_cache


def _cameras_invalidate() -> None:
    """Задайте cache = None → next access ще зареди от DB."""
    global _cameras_cache
    _cameras_cache = None


def _get_camera(ch: int) -> dict | None:
    """Бърз lookup от cache. Връща None ако няма."""
    return _cameras_get_all().get(ch)


def _db_insert_detection(
    ch: int, detected_at: str, labels: list, confs: list, img_path: str,
    boxes: list | None = None, tracks: list | None = None,
) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO detection_events(channel,detected_at,labels,confidences,image_path,boxes,tracks) "
            "VALUES(?,?,?,?,?,?,?)",
            (ch, detected_at, json.dumps(labels), json.dumps(confs), img_path,
             json.dumps(boxes) if boxes else None,
             json.dumps(tracks) if tracks else None),
        )


def _db_delete_old_detections(retention_days: int) -> None:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT image_path FROM detection_events WHERE detected_at < ?", (cutoff,)
        ).fetchall()
        for (p,) in rows:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass
        conn.execute("DELETE FROM detection_events WHERE detected_at < ?", (cutoff,))


def _db_trim_events(max_count: int = 1000) -> int:
    """Изтрива най-старите events ако общият брой надвиши max_count. Връща изтритите."""
    with sqlite3.connect(DB_PATH) as conn:
        total = conn.execute("SELECT COUNT(*) FROM detection_events").fetchone()[0]
        if total <= max_count:
            return 0
        to_del = total - max_count
        rows = conn.execute(
            "SELECT id, image_path FROM detection_events ORDER BY detected_at ASC LIMIT ?",
            (to_del,),
        ).fetchall()
        for (eid, p) in rows:
            try:
                if p:
                    Path(p).unlink(missing_ok=True)
            except Exception:
                pass
            conn.execute("DELETE FROM detection_events WHERE id=?", (eid,))
        return to_del


# ── Face recognition DB helpers ────────────────────────────────────────────

def _db_meta_get(key: str, default: str = "") -> str:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default


def _db_meta_set(key: str, value: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (key, value))


def _db_get_cluster_representatives() -> list[dict]:
    """Връща id + embedding на всички клъстери (за сравнение)."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, embedding FROM face_clusters WHERE embedding IS NOT NULL"
        ).fetchall()
        return [{"id": r[0], "embedding": json.loads(r[1])} for r in rows]


# Максимум face crops на клъстер: достатъчно за разнообразие на ъгли/осветление,
# спира безкрайното нарастване на DB и ускорява сравнението на нови лица.
MAX_CROPS_PER_CLUSTER: int = 20


def _db_get_cluster_crop_count(cluster_id: int) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM face_crops WHERE cluster_id=?", (cluster_id,),
        ).fetchone()[0]


def _db_get_all_cluster_embeddings() -> dict[int, list]:
    """Връща {cluster_id: [embedding1, embedding2, ...]} от ВСИЧКИ face_crops.

    Това позволява сравнение на ново лице срещу всяка снимка в клъстера
    (не само representative), което прави системата self-learning при merge
    или при натрупване на разнообразни ъгли/осветления на едно и също лице.
    """
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT cluster_id, embedding FROM face_crops WHERE cluster_id IS NOT NULL"
        ).fetchall()
    result: dict[int, list] = {}
    for cid, emb_json in rows:
        try:
            result.setdefault(cid, []).append(json.loads(emb_json))
        except Exception:
            pass
    return result


def _db_get_all_cluster_embeddings_v2() -> dict[int, list]:
    """Връща {cluster_id: [v2_emb1, v2_emb2, ...]} от ВСИЧКИ face_crops с v2 embedding.

    InsightFace ArcFace 512-d embeddings (срещу 128-d на dlib).
    """
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT cluster_id, embedding_v2 FROM face_crops "
            "WHERE cluster_id IS NOT NULL AND embedding_v2 IS NOT NULL"
        ).fetchall()
    result: dict[int, list] = {}
    for cid, emb_json in rows:
        try:
            result.setdefault(cid, []).append(json.loads(emb_json))
        except Exception:
            pass
    return result


def _db_get_crops_without_v2() -> list[tuple[int, str, int]]:
    """Връща [(crop_id, image_path, cluster_id), ...] за crops без v2 embedding."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, image_path, cluster_id FROM face_crops "
            "WHERE embedding_v2 IS NULL OR embedding_v2 = ''"
        ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def _db_set_crop_embedding_v2(crop_id: int, embedding_v2: list) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE face_crops SET embedding_v2=? WHERE id=?",
            (json.dumps(embedding_v2), crop_id),
        )


def _db_create_cluster(created_at: str, embedding: list, embedding_v2: list | None = None) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "INSERT INTO face_clusters(created_at, face_count, embedding, embedding_v2) "
            "VALUES(?,1,?,?)",
            (created_at, json.dumps(embedding) if embedding else None,
             json.dumps(embedding_v2) if embedding_v2 else None),
        )
        return cur.lastrowid


def _db_update_cluster_count(cluster_id: int) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE face_clusters SET face_count = face_count + 1 WHERE id=?",
            (cluster_id,),
        )


def _db_update_cluster_representative(cluster_id: int, img_path: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE face_clusters SET representative=? WHERE id=? AND representative IS NULL",
            (img_path, cluster_id),
        )


def _db_insert_face_crop(
    detected_at: str, channel: int, image_path: str,
    embedding: list, cluster_id: int, event_id: int,
    embedding_v2: list | None = None,
) -> None:
    """Записва face crop. embedding е dlib 128-d (legacy), embedding_v2 е InsightFace 512-d."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO face_crops(detected_at,channel,image_path,embedding,cluster_id,event_id,embedding_v2) "
            "VALUES(?,?,?,?,?,?,?)",
            (detected_at, channel, image_path,
             json.dumps(embedding) if embedding else "[]",
             cluster_id, event_id,
             json.dumps(embedding_v2) if embedding_v2 else None),
        )


def _db_update_event_faces(event_id: int, faces: list[dict]) -> None:
    """Записва разпознатите лица в detection_events.faces.

    faces: list of {cluster_id, name} — name може да бъде None за непознати.
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE detection_events SET faces=? WHERE id=?",
            (json.dumps(faces, ensure_ascii=False), event_id),
        )


def _db_get_cluster_names(cluster_ids: list[int]) -> dict[int, str | None]:
    if not cluster_ids:
        return {}
    placeholders = ",".join("?" * len(cluster_ids))
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            f"SELECT id, name FROM face_clusters WHERE id IN ({placeholders})",
            cluster_ids,
        ).fetchall()
        return {r[0]: r[1] for r in rows}


def _db_get_new_person_events(since_id: int) -> list[dict]:
    """Връща detection_events в които има 'person' в кадъра, след since_id.

    След въвеждането на tracking, `labels` колоната съдържа само НОВИТЕ tracks
    в това event, докато `boxes` пази ВСИЧКИ детекции в кадъра. Затова филтрираме
    по boxes (за нови events) ИЛИ labels (за стари events отпреди tracking-а
    или за events от ingest endpoints без boxes).
    """
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, channel, detected_at, labels, image_path, boxes "
            "FROM detection_events "
            "WHERE id > ? AND (boxes LIKE ? OR labels LIKE ?) "
            "ORDER BY id ASC LIMIT 50",
            (since_id, '%"label": "person"%', '%"person"%'),
        ).fetchall()
        return [
            {"id": r[0], "channel": r[1], "detected_at": r[2],
             "labels": r[3], "image_path": r[4], "boxes": r[5]}
            for r in rows
        ]


def _db_list_clusters(include_unnamed: bool = True) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        q = "SELECT id,name,created_at,face_count,representative FROM face_clusters"
        if not include_unnamed:
            q += " WHERE name IS NOT NULL"
        q += " ORDER BY face_count DESC, id DESC"
        rows = conn.execute(q).fetchall()
        return [
            {"id": r[0], "name": r[1], "created_at": r[2],
             "face_count": r[3], "representative": r[4]}
            for r in rows
        ]


def _db_get_cluster_crops(cluster_id: int) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, detected_at, channel, image_path FROM face_crops "
            "WHERE cluster_id=? ORDER BY detected_at DESC",
            (cluster_id,),
        ).fetchall()
        return [
            {"id": r[0], "detected_at": r[1], "channel": r[2], "image_path": r[3]}
            for r in rows
        ]


def _db_name_cluster(cluster_id: int, name: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE face_clusters SET name=? WHERE id=?", (name, cluster_id))


def _db_propagate_cluster_name(cluster_id: int, name: str | None) -> int:
    """Обновява detection_events.faces с новото име навсякъде, където този cluster присъства."""
    needle = f'"cluster_id": {cluster_id}'
    updated = 0
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, faces FROM detection_events WHERE faces LIKE ?",
            (f"%{needle}%",),
        ).fetchall()
        for ev_id, faces_json in rows:
            try:
                faces = json.loads(faces_json)
            except Exception:
                continue
            changed = False
            for f in faces:
                if f.get("cluster_id") == cluster_id and f.get("name") != name:
                    f["name"] = name
                    changed = True
            if changed:
                conn.execute(
                    "UPDATE detection_events SET faces=? WHERE id=?",
                    (json.dumps(faces, ensure_ascii=False), ev_id),
                )
                updated += 1
    return updated


def _db_merge_clusters(src_id: int, dst_id: int) -> None:
    """Премества всички crops от src → dst, обновява event.faces, изтрива src."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE face_crops SET cluster_id=? WHERE cluster_id=?", (dst_id, src_id)
        )
        # Обновяваме face_count на dst
        count = conn.execute(
            "SELECT COUNT(*) FROM face_crops WHERE cluster_id=?", (dst_id,)
        ).fetchone()[0]
        dst_name_row = conn.execute(
            "SELECT name FROM face_clusters WHERE id=?", (dst_id,)
        ).fetchone()
        dst_name = dst_name_row[0] if dst_name_row else None
        conn.execute(
            "UPDATE face_clusters SET face_count=? WHERE id=?", (count, dst_id)
        )
        conn.execute("DELETE FROM face_clusters WHERE id=?", (src_id,))

        # Обновяваме detection_events.faces — заменяме src_id със dst_id и пренасяме името
        needle = f'"cluster_id": {src_id}'
        rows = conn.execute(
            "SELECT id, faces FROM detection_events WHERE faces LIKE ?",
            (f"%{needle}%",),
        ).fetchall()
        for ev_id, faces_json in rows:
            try:
                faces = json.loads(faces_json)
            except Exception:
                continue
            new_faces: list[dict] = []
            seen_cids: set[int] = set()
            for f in faces:
                cid = dst_id if f.get("cluster_id") == src_id else f.get("cluster_id")
                if cid in seen_cids:
                    continue
                seen_cids.add(cid)
                if cid == dst_id:
                    f["cluster_id"] = dst_id
                    f["name"]       = dst_name
                new_faces.append(f)
            conn.execute(
                "UPDATE detection_events SET faces=? WHERE id=?",
                (json.dumps(new_faces, ensure_ascii=False), ev_id),
            )


def _db_delete_cluster(cluster_id: int) -> list[str]:
    """Изтрива клъстер и всички face_crops му. Връща пътищата за изтриване."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT image_path FROM face_crops WHERE cluster_id=?", (cluster_id,)
        ).fetchall()
        paths = [r[0] for r in rows]
        conn.execute("DELETE FROM face_crops WHERE cluster_id=?", (cluster_id,))
        conn.execute("DELETE FROM face_clusters WHERE id=?", (cluster_id,))
        return paths


# ── Face extraction logic ──────────────────────────────────────────────────
#
# Двустепенна архитектура:
#   v2 (InsightFace ArcFace) — primary: 512-d embeddings, cosine similarity, ~99.8% LFW
#   v1 (dlib HOG/CNN) — legacy fallback за случаи когато InsightFace не е достъпен
#
# При startup _migrate_faces_to_v2() пресмята v2 embeddings за всички съществуващи
# crop файлове, така че named clusters (Алек, Нана, ...) да продължат да работят.
# ────────────────────────────────────────────────────────────────────────────

_insightface_app = None
_insightface_lock = threading.Lock()
_insightface_status = "uninitialized"  # "ready", "failed", "uninitialized"

# Cosine similarity thresholds за ArcFace ArcFace embeddings (L2 normalized).
# 1 = идентични, 0 = ортогонални. Препоръка от ArcFace paper:
#   > 0.50 = same person (висока сигурност)
#   0.40-0.50 = same person с margin check
#   < 0.40 = вероятно друг човек
ARCFACE_STRICT_THRESHOLD = 0.50
ARCFACE_RELAXED_THRESHOLD = 0.40
ARCFACE_MARGIN = 0.05


def _get_insightface_app():
    """Lazy-initialize InsightFace FaceAnalysis (singleton, thread-safe)."""
    global _insightface_app, _insightface_status
    if _insightface_app is not None:
        return _insightface_app
    with _insightface_lock:
        if _insightface_app is not None:
            return _insightface_app
        try:
            import insightface
            app = insightface.app.FaceAnalysis(
                name="buffalo_l",
                root=os.environ.get("INSIGHTFACE_HOME", "/app/insightface_models"),
                providers=["CPUExecutionProvider"],
                allowed_modules=["detection", "recognition"],
            )
            # det_size: по-голямо = повече лица намерени, но по-бавно.
            # 640x640 е default, добър баланс. Можем да упоменем по-голямо
            # за HD кадри ако трябва.
            app.prepare(ctx_id=-1, det_size=(640, 640))
            _insightface_app = app
            _insightface_status = "ready"
            _LOGGER.info("👤 InsightFace buffalo_l: заредeн ✓ (SCRFD detection + ArcFace 512-d recognition)")
        except Exception as exc:
            _insightface_status = "failed"
            _LOGGER.warning(
                "InsightFace not available — fallback to dlib HOG/CNN: %s", exc,
            )
            _insightface_app = None
    return _insightface_app


def _cosine_similarity(a, b):
    """Cosine similarity между две L2-нормализирани embedding-а (ArcFace са L2 norm)."""
    import numpy as _np
    a_arr = _np.array(a, dtype=_np.float32)
    b_arr = _np.array(b, dtype=_np.float32)
    return float(_np.dot(a_arr, b_arr) / (_np.linalg.norm(a_arr) * _np.linalg.norm(b_arr) + 1e-9))


def _find_or_create_cluster_v2(embedding_v2: list) -> int:
    """ArcFace 512-d clustering — primary path.

    Cosine similarity (вместо Euclidean distance) за нормализирани ArcFace embeddings.
    Двустепенна стратегия (като v1):
      1. Strict: cos_sim >= 0.50 → приема веднага
      2. Confident: cos_sim >= 0.40 И margin (top-1 - top-2) >= 0.05 → приема
    """
    cluster_embs_v2 = _db_get_all_cluster_embeddings_v2()
    if cluster_embs_v2:
        try:
            scored: list[tuple[int, float]] = []
            for cid, embs in cluster_embs_v2.items():
                # Best (max) cosine similarity срещу която и да е embedding в клъстера
                best_sim = max(_cosine_similarity(embedding_v2, e) for e in embs)
                scored.append((cid, best_sim))
            scored.sort(key=lambda x: -x[1])  # descending similarity

            best_cid, best_sim = scored[0]
            second_sim = scored[1][1] if len(scored) >= 2 else 0.0
            margin = best_sim - second_sim

            if best_sim >= ARCFACE_STRICT_THRESHOLD:
                _db_update_cluster_count(best_cid)
                _LOGGER.debug("Face v2: STRICT match cluster #%d (cos=%.3f)", best_cid, best_sim)
                return best_cid

            if best_sim >= ARCFACE_RELAXED_THRESHOLD and margin >= ARCFACE_MARGIN:
                _db_update_cluster_count(best_cid)
                _LOGGER.info(
                    "Face v2: CONFIDENT match cluster #%d (cos=%.3f, margin=%.3f над #%d)",
                    best_cid, best_sim, margin, scored[1][0],
                )
                return best_cid
        except Exception as exc:
            _LOGGER.warning("Face v2 cluster match: %s", exc)

    # Нов клъстер — записваме v2 embedding (v1 ще е празно)
    now = datetime.now(timezone.utc).isoformat()
    return _db_create_cluster(created_at=now, embedding=[], embedding_v2=embedding_v2)


def _find_or_create_cluster(encoding: list) -> int:
    """Намира клъстер чрез min-distance към ВСЯКА снимка в клъстера или създава нов.

    Двустепенна стратегия:
      1. Strict match (dist <= 0.50) — приема ВЕДНАГА (висока сигурност)
      2. Confident match (dist <= 0.55 И top-1 поне 0.04 по-добро от top-2) —
         приема при clear winner. Ако top-1 и top-2 са близо (gap < 0.04),
         не рискуваме false positive → нов клъстер.

    face_recognition стандартен threshold е 0.6, но нашето strict 0.50 + relaxed
    0.55-with-margin дава добър баланс — намаляваме false negatives (Нана case-а
    с dist=0.501 вече минава) без значителни false positives.

    Self-learning: всяка нова снимка става нова "проба" — следващите сравнения
    я виждат, така че при merge клъстерите автоматично "наследяват" разпознаването.
    """
    import numpy as _np

    cluster_embs = _db_get_all_cluster_embeddings()
    if cluster_embs:
        try:
            import face_recognition as _fr
            target = _np.array(encoding)
            scored: list[tuple[int, float]] = []
            for cid, embs in cluster_embs.items():
                dists = _fr.face_distance([_np.array(e) for e in embs], target)
                scored.append((cid, float(dists.min())))
            scored.sort(key=lambda x: x[1])

            best_cid, best_dist = scored[0]
            second_dist = scored[1][1] if len(scored) >= 2 else 1.0
            margin = second_dist - best_dist

            # Step 1: strict match — висока сигурност
            if best_dist <= 0.50:
                _db_update_cluster_count(best_cid)
                _LOGGER.debug("Face: STRICT match cluster #%d (dist=%.3f)", best_cid, best_dist)
                return best_cid

            # Step 2: confident match — top-1 е "clear winner" над top-2
            if best_dist <= 0.55 and margin >= 0.04:
                _db_update_cluster_count(best_cid)
                _LOGGER.info(
                    "Face: CONFIDENT match cluster #%d (dist=%.3f, margin=%.3f над #%d)",
                    best_cid, best_dist, margin, scored[1][0],
                )
                return best_cid
        except Exception as exc:
            _LOGGER.debug("Face cluster match: %s", exc)

    now = datetime.now(timezone.utc).isoformat()
    return _db_create_cluster(created_at=now, embedding=encoding)


def _extract_faces_insightface(ev: dict) -> int:
    """InsightFace-based face extraction (primary path).

    Един проход — SCRFD detection + ArcFace recognition в една стъпка.
    SCRFD detection AP: 95% (WIDERFACE Hard) — 1.5× по-добро от dlib HOG (~70%).
    ArcFace recognition: 99.83% LFW (vs 99.38% dlib) → ~2.7× по-малко false matches.

    На 1080p CPU: ~1-2 сек total. На 1440p: ~3-4 сек.
    """
    import cv2
    import numpy as _np
    from PIL import Image as _PilImg

    app = _get_insightface_app()
    if app is None:
        # Fallback към dlib ако InsightFace fail-нал да се зареди
        return _extract_faces_dlib(ev)

    img_path = Path(ev["image_path"])
    if not img_path.exists():
        return 0

    # InsightFace очаква BGR (OpenCV format)
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        _LOGGER.warning("Face InsightFace: cannot load image %s", img_path)
        return 0

    img_h, img_w = img_bgr.shape[:2]
    try:
        faces = app.get(img_bgr)
    except Exception as exc:
        _LOGGER.warning("Face InsightFace inference event #%d: %s", ev["id"], exc)
        return 0

    if not faces:
        return 0

    count = 0
    found_clusters: list[int] = []

    for i, face in enumerate(faces):
        # Detection score (0-1) — фарираме фалшиви детекции
        if face.det_score < 0.5:
            continue
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        fw, fh = x2 - x1, y2 - y1
        if fw < 24 or fh < 24:
            continue

        embedding_v2 = face.embedding.tolist()  # 512-d
        cluster_id = _find_or_create_cluster_v2(embedding_v2)
        found_clusters.append(cluster_id)
        count += 1

        # Cap на crop файлове per cluster
        existing = _db_get_cluster_crop_count(cluster_id)
        if existing >= MAX_CROPS_PER_CLUSTER:
            continue

        # Запазваме crop файл (с padding)
        margin = max(8, int(min(fh, fw) * 0.2))
        cx1 = max(0, x1 - margin); cy1 = max(0, y1 - margin)
        cx2 = min(img_w, x2 + margin); cy2 = min(img_h, y2 + margin)
        face_arr = img_bgr[cy1:cy2, cx1:cx2]

        cluster_dir = FACES_DIR / str(cluster_id)
        cluster_dir.mkdir(parents=True, exist_ok=True)
        ts_str    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        crop_path = cluster_dir / f"{ts_str}_{i}.jpg"
        # cv2 save (BGR директно)
        try:
            cv2.imwrite(str(crop_path), face_arr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        except Exception:
            # Fallback с PIL (RGB conversion)
            face_rgb = cv2.cvtColor(face_arr, cv2.COLOR_BGR2RGB)
            _PilImg.fromarray(face_rgb).save(str(crop_path), "JPEG", quality=90)

        _db_insert_face_crop(
            detected_at=ev["detected_at"],
            channel=ev["channel"],
            image_path=str(crop_path),
            embedding=[],  # v1 dlib празно за нови crops от InsightFace
            cluster_id=cluster_id,
            event_id=ev["id"],
            embedding_v2=embedding_v2,
        )
        _db_update_cluster_representative(cluster_id, str(crop_path))
        _LOGGER.debug(
            "Face v2: клъстер #%d crop %d/%d (event %d, det_score=%.2f, %dx%d)",
            cluster_id, existing + 1, MAX_CROPS_PER_CLUSTER, ev["id"],
            face.det_score, fw, fh,
        )

    if found_clusters:
        unique_cids = list(dict.fromkeys(found_clusters))
        names_map   = _db_get_cluster_names(unique_cids)
        faces_meta  = [
            {"cluster_id": cid, "name": names_map.get(cid)}
            for cid in unique_cids
        ]
        _db_update_event_faces(ev["id"], faces_meta)

    return count


def _extract_faces_from_event(ev: dict) -> int:
    """Wrapper: ползва InsightFace ако е достъпен, fallback към dlib HOG/CNN."""
    if _insightface_status == "ready" or (
        _insightface_status == "uninitialized" and _get_insightface_app() is not None
    ):
        return _extract_faces_insightface(ev)
    return _extract_faces_dlib(ev)


def _extract_faces_dlib(ev: dict) -> int:
    """LEGACY dlib HOG/CNN face extraction (fallback ако InsightFace не работи)."""
    import face_recognition as _fr
    import numpy as _np
    from PIL import Image as _PilImg

    img_path = Path(ev["image_path"])
    if not img_path.exists():
        return 0

    try:
        full_image = _fr.load_image_file(str(img_path))
    except Exception:
        return 0

    img_h, img_w = full_image.shape[:2]
    count = 0
    seen_face_centers: list[tuple[int, int]] = []   # за дедупликация между full + crops
    found_clusters: list[int] = []                  # cluster_id-та намерени в този event

    def _process_locations(crop: "_np.ndarray", locs: list, ox: int, oy: int, scale: float) -> int:
        """Обработва намерени face locations в region. Връща брой нови запазени лица."""
        nonlocal count, seen_face_centers, found_clusters
        if not locs:
            return 0
        encs = _fr.face_encodings(crop, locs)
        added = 0
        for i, (location, encoding) in enumerate(zip(locs, encs)):
            top, right, bottom, left = location
            fh, fw = bottom - top, right - left
            # Минимална размер: 24px (снижено от 30px за да хващаме по-далечни лица).
            # face_recognition encoding-ът работи разумно дори за 24x24 лица.
            if fh < 24 or fw < 24:
                continue

            # Конвертираме координатите към оригиналното изображение (за дедупликация)
            orig_cx = int(ox + (left + right) / 2 / scale)
            orig_cy = int(oy + (top + bottom) / 2 / scale)
            # Пропусни ако вече сме обработили лице в радиус 50px (вече намерено от друг region)
            if any(abs(orig_cx - cx) < 50 and abs(orig_cy - cy) < 50
                   for cx, cy in seen_face_centers):
                continue
            seen_face_centers.append((orig_cx, orig_cy))

            margin  = max(8, int(min(fh, fw) * 0.2))
            t2 = max(0, top - margin); l2 = max(0, left - margin)
            b2 = min(crop.shape[0], bottom + margin)
            r2 = min(crop.shape[1], right + margin)
            face_arr = crop[t2:b2, l2:r2]

            cluster_id = _find_or_create_cluster(encoding.tolist())
            found_clusters.append(cluster_id)
            count += 1

            # Cap: записваме нов crop файл само ако клъстерът има < MAX_CROPS_PER_CLUSTER.
            # Това дава разнообразие за разпознаване, но избягва безкрайно нарастване
            # на DB/диск при чест посетител. face_count в face_clusters все пак нараства
            # (от _find_or_create_cluster) → отразява "брой пъти видян".
            existing = _db_get_cluster_crop_count(cluster_id)
            if existing >= MAX_CROPS_PER_CLUSTER:
                _LOGGER.debug(
                    "Face: клъстер #%d пълен (%d/%d) — прескачам запис на crop",
                    cluster_id, existing, MAX_CROPS_PER_CLUSTER,
                )
                added += 1
                continue

            cluster_dir = FACES_DIR / str(cluster_id)
            cluster_dir.mkdir(parents=True, exist_ok=True)
            ts_str    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            crop_path = cluster_dir / f"{ts_str}_{i}.jpg"
            _PilImg.fromarray(face_arr).save(str(crop_path), "JPEG", quality=90)

            _db_insert_face_crop(
                detected_at=ev["detected_at"],
                channel=ev["channel"],
                image_path=str(crop_path),
                embedding=encoding.tolist(),
                cluster_id=cluster_id,
                event_id=ev["id"],
            )
            _db_update_cluster_representative(cluster_id, str(crop_path))
            added += 1
            _LOGGER.debug("Face: клъстер #%d crop %d/%d (event %d)",
                          cluster_id, existing + 1, MAX_CROPS_PER_CLUSTER, ev["id"])
        return added

    # 1. Винаги пробваме face detection на пълното изображение (HOG x1 е бърз ~0.3s).
    #    YOLO може да пропусне човек дори при ясна снимка (както в case ev1404).
    full_locs = _fr.face_locations(full_image, model="hog", number_of_times_to_upsample=1)
    _process_locations(full_image, full_locs, 0, 0, 1.0)

    # 1b. Ако HOG x1 не намери НИЩО, опитваме HOG x2 на full image (по-чувствителен,
    #     но 4× по-бавен). Това хваща допълнителни ~30% от случаите с малки/нечетливи
    #     лица (тестове показват: ev2528 hit само на x2). Само ако x1 fail-нал —
    #     иначе х1 е достатъчен.
    if not full_locs:
        full_locs = _fr.face_locations(full_image, model="hog", number_of_times_to_upsample=2)
        _process_locations(full_image, full_locs, 0, 0, 1.0)

    # 2. Допълнително: per-person upscaled crops (за далечни хора с малки лица).
    #    Target минимална резолюция на crop: 600px (вместо 300) — face area вътре
    #    в crop-а обикновено е ~10-15% от височината, така че по-голям upscale
    #    значи по-голямо/четливо лице за HOG.
    raw_boxes = json.loads(ev["boxes"]) if ev.get("boxes") else []
    person_boxes = [b["box"] for b in raw_boxes if b.get("label") == "person"]

    crop_found_any = bool(full_locs)
    for box in person_boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        pad = max(10, int((y2 - y1) * 0.1))
        x1c = max(0, x1 - pad); y1c = max(0, y1 - pad)
        x2c = min(img_w, x2 + pad); y2c = min(img_h, y2 + pad)
        crop = full_image[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            continue
        cropped_h, cropped_w = crop.shape[:2]
        scale = max(1.0, 600.0 / min(cropped_h, cropped_w))
        if scale > 1.0:
            new_w, new_h = int(cropped_w * scale), int(cropped_h * scale)
            crop = _np.array(
                _PilImg.fromarray(crop).resize((new_w, new_h), _PilImg.LANCZOS)
            )
        locs = _fr.face_locations(crop, model="hog", number_of_times_to_upsample=1)
        added = _process_locations(crop, locs, x1c, y1c, scale)
        if added:
            crop_found_any = True

    # 3. CNN fallback — само ако HOG не е намерил НИЩО досега.
    #    CNN е бавен (4-15s в зависимост от резолюцията) но е значително по-добър
    #    за лица под ъгъл, странични профили, и нечетливи кадри (Shelly Cam 1
    #    показва точно такива случаи). Изпълнява се на full image без upsample
    #    за speed. Очакван overhead: ~5-8s на Shelly 1920x1080.
    if not crop_found_any:
        try:
            cnn_locs = _fr.face_locations(full_image, model="cnn", number_of_times_to_upsample=0)
            if cnn_locs:
                _LOGGER.info(
                    "👤 Face: CNN fallback намери %d лице(а) в event #%d (HOG fail-на)",
                    len(cnn_locs), ev["id"],
                )
                _process_locations(full_image, cnn_locs, 0, 0, 1.0)
        except Exception as exc:
            _LOGGER.warning("Face CNN fallback event #%d: %s", ev["id"], exc)

    # Записваме разпознатите лица в event-а (имена + cluster_id)
    if found_clusters:
        unique_cids = list(dict.fromkeys(found_clusters))
        names_map   = _db_get_cluster_names(unique_cids)
        faces_meta  = [
            {"cluster_id": cid, "name": names_map.get(cid)}
            for cid in unique_cids
        ]
        _db_update_event_faces(ev["id"], faces_meta)

    return count


def _db_get_config(channels: list[int]) -> dict[int, dict]:
    """Зарежда arm + min_confidence от SQLite. Default: armed=True, min_conf=None."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT channel, armed, min_confidence FROM detection_config"
        ).fetchall()
    cfg = {ch: {"armed": True, "min_confidence": None} for ch in channels}
    for ch, armed, mc in rows:
        cfg[ch] = {"armed": bool(armed), "min_confidence": mc}
    return cfg


# Backward compat alias
def _db_get_armed(channels: list[int]) -> dict[int, bool]:
    return {ch: v["armed"] for ch, v in _db_get_config(channels).items()}


def _db_set_armed(channel: int, armed: bool) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO detection_config(channel, armed) VALUES(?,?) "
            "ON CONFLICT(channel) DO UPDATE SET armed=excluded.armed",
            (channel, int(armed)),
        )


def _db_set_min_confidence(channel: int, min_conf: float | None) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO detection_config(channel, min_confidence) VALUES(?,?) "
            "ON CONFLICT(channel) DO UPDATE SET min_confidence=excluded.min_confidence",
            (channel, min_conf),
        )


def _db_get_detections(channel: int | None, limit: int) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        if channel is not None:
            rows = conn.execute(
                "SELECT * FROM detection_events WHERE channel=? "
                "ORDER BY detected_at DESC LIMIT ?",
                (channel, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM detection_events ORDER BY detected_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["labels"]      = json.loads(d["labels"])
        d["confidences"] = json.loads(d["confidences"])
        if d.get("faces"):
            try:
                d["faces"] = json.loads(d["faces"])
            except Exception:
                d["faces"] = []
        else:
            d["faces"] = []
        # Изваждаме броя обекти в кадъра от boxes (всички детекции, не само нови tracks).
        # След tracking, labels съдържа само НОВИТЕ tracks за това event.
        # boxes пази ВСИЧКИ детекции в кадъра — оттам взимаме истинския count.
        try:
            raw_boxes = json.loads(d.get("boxes") or "[]")
        except Exception:
            raw_boxes = []
        frame_counts: dict[str, int] = {}
        for b in raw_boxes:
            if isinstance(b, dict) and b.get("label"):
                frame_counts[b["label"]] = frame_counts.get(b["label"], 0) + 1
        d["frame_counts"] = frame_counts  # {"person": 3, "car": 1, ...}
        d.pop("boxes", None)
        if d.get("tracks"):
            try:
                d["tracks"] = json.loads(d["tracks"])
            except Exception:
                d["tracks"] = []
        else:
            d["tracks"] = []
        result.append(d)
    return result


def _db_insert(d: dict) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """INSERT INTO readings
               (ts, water_in, water_out, ambient, target, hvac_mode, preset, power_on, errors)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                int(datetime.now(timezone.utc).timestamp()),
                d.get("water_in_c"),
                d.get("water_out_c"),
                d.get("ambient_c"),
                d.get("target_c"),
                d.get("hvac_mode"),
                d.get("preset"),
                int(bool(d.get("power_on"))),
                d.get("errors"),
            ),
        )


def _db_insert_deye(d: dict) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """INSERT INTO deye_readings
               (ts, total_power_w, battery_power_w, pv1_power_w, grid_power_w,
                battery_soc_pct, pv1_voltage_v, grid_voltage_v, current_l1_a, ac_freq_hz)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                int(datetime.now(timezone.utc).timestamp()),
                d.get("total_power_w"),
                d.get("battery_power_w"),
                d.get("pv1_power_w"),
                d.get("grid_power_w"),
                d.get("battery_soc_pct"),
                d.get("pv1_voltage_v"),
                d.get("grid_voltage_v"),
                d.get("current_l1_a"),
                d.get("ac_freq_hz"),
            ),
        )


def _db_history(hours: int = 24) -> list[dict]:
    since = int(datetime.now(timezone.utc).timestamp()) - hours * 3600
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM readings WHERE ts >= ? ORDER BY ts ASC", (since,)
        ).fetchall()
    return [dict(r) for r in rows]


def _db_deye_history(hours: int = 24) -> list[dict]:
    since = int(datetime.now(timezone.utc).timestamp()) - hours * 3600
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM deye_readings WHERE ts >= ? ORDER BY ts ASC", (since,)
        ).fetchall()
    return [dict(r) for r in rows]


def _db_get_last_ev_total() -> float:
    """Връща последното запазено total_kwh или 0.0."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT total_kwh FROM ev_readings ORDER BY ts DESC LIMIT 1"
            ).fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0
    except Exception:
        return 0.0


def _db_insert_ev(d: dict) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """INSERT INTO ev_readings (ts, current_a, power_w, session_kwh, total_kwh, state)
               VALUES (?,?,?,?,?,?)""",
            (
                int(datetime.now(timezone.utc).timestamp()),
                d.get("current_a"),
                d.get("power_w"),
                d.get("session_kwh"),
                d.get("total_kwh"),
                d.get("state"),
            ),
        )


def _db_ev_history(hours: int = 24) -> list[dict]:
    since = int(datetime.now(timezone.utc).timestamp()) - hours * 3600
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM ev_readings WHERE ts >= ? ORDER BY ts ASC", (since,)
        ).fetchall()
    return [dict(r) for r in rows]


# ── Detection worker ───────────────────────────────────────────────────────
async def _detection_loop() -> None:
    """Background task: YOLO inference върху вече кешираните snapshot-и (0 допълнителни заявки към NVR)."""
    if not nvr_settings.host or not detect_settings.enabled:
        return

    loop = asyncio.get_running_loop()

    try:
        import io as _bio
        import numpy as _np
        from PIL import Image as _PilImage
        from ultralytics import YOLO as _YOLO
    except ImportError:
        _LOGGER.error("ultralytics/pillow не са инсталирани — detection е изключен")
        return

    _LOGGER.info("Detection: зарежда %s модел…", detect_settings.model)
    model = await loop.run_in_executor(None, lambda: _YOLO(detect_settings.model))
    _LOGGER.info("Detection: модел зареден ✓ (%s)", detect_settings.model)

    detect_cls  = [int(x.strip()) for x in detect_settings.classes.split(",") if x.strip()]
    conf_thresh = detect_settings.confidence
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Зарежда arm статус от DB (default=True за всички — включва и Extra IP cams)
    channels = _all_valid_channels()
    cam_cfg = await asyncio.to_thread(_db_get_config, channels)
    global _detect_armed, _detect_conf
    for ch, v in cam_cfg.items():
        _detect_armed[ch] = v["armed"]
        _detect_conf[ch]  = v["min_confidence"]

    last_event:      dict[int, float] = {}   # throttle
    last_snap_ts:    dict[int, float] = {}   # последно обработен snapshot ts
    last_cleanup:    float = time.monotonic()
    auth = httpx.DigestAuth(nvr_settings.user, nvr_settings.password)

    async with httpx.AsyncClient(timeout=httpx.Timeout(8.0)) as client:
        while True:
            # Почистване на стари events веднъж на час
            if time.monotonic() - last_cleanup > 3600:
                await asyncio.to_thread(_db_delete_old_detections, detect_settings.retention)
                last_cleanup = time.monotonic()

            for ch in _all_valid_channels():
                # Пропускаме DISARMED камери
                if not _detect_armed.get(ch, True):
                    continue

                try:
                    now_m  = time.monotonic()
                    cached = _nvr_snap_cache.get(ch)
                    extra  = _extra_cam(ch)

                    # Extra cams с snap_url: по-кратък cache (HTTP е бърз ~2s)
                    # Extra cams без snap_url: по-дълъг cache (RTSP grab е бавен)
                    has_http_snap = extra and extra.get("snap_url")
                    eff_cache_s = (5 if has_http_snap else 15) if extra else nvr_settings.cache_s

                    if cached and (now_m - cached[0]) < eff_cache_s:
                        # Кешът е пресен — ползваме го (избягваме нов snapshot fetch)
                        snap_ts, jpeg = cached
                    elif _hls_latest_ts(ch) is not None:
                        # ── HLS .ts grab (preferred): четем кадър от вече-recorded
                        # HLS segment вместо да отваряме нов RTSP/HTTP connection.
                        # ~16x по-бързо, и НЕ натоварва камерата/NVR-a с extra връзка.
                        jpeg = b""
                        ts_path = _hls_latest_ts(ch)
                        grab_cmd = [
                            "ffmpeg", "-y",
                            "-i", str(ts_path),
                            "-an",
                            "-vframes", "1",
                            "-q:v", "3",
                            "-update", "1",
                            "-f", "image2",
                            "pipe:1",
                        ]
                        try:
                            grab_proc = await asyncio.create_subprocess_exec(
                                *grab_cmd,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE,
                            )
                            grab_out, _ = await asyncio.wait_for(grab_proc.communicate(), timeout=4)
                            if grab_out and len(grab_out) > 3_000:
                                jpeg = grab_out
                        except asyncio.TimeoutError:
                            _LOGGER.warning("Detection cam%d: HLS .ts grab timeout", ch)
                            try:
                                grab_proc.kill()
                            except Exception:
                                pass
                        except Exception as exc:
                            _LOGGER.debug("Detection cam%d: HLS .ts grab fail: %s", ch, exc)
                        if not jpeg:
                            continue
                        snap_ts = now_m
                        _nvr_snap_cache[ch] = (now_m, jpeg)
                    elif extra:
                        # Extra IP camera (Reolink/IMOU): при наличен snap_url ползваме
                        # директния HTTP API (1-2s, full-res), иначе RTSP grab от sub-stream.
                        jpeg = b""
                        snap_url = extra.get("snap_url")
                        if snap_url:
                            try:
                                resp = await asyncio.wait_for(
                                    client.get(snap_url), timeout=5
                                )
                                if resp.status_code == 200 and len(resp.content) > 3_000:
                                    jpeg = resp.content
                            except Exception as exc:
                                _LOGGER.debug("Detection cam%d (extra): HTTP snap fail: %s", ch, exc)
                        if not jpeg:
                            # Fallback: RTSP frame grab
                            rtsp_grab = extra.get("rtsp_sub") or extra.get("rtsp_main")
                            grab_cmd = [
                                "ffmpeg", "-y",
                                "-rtsp_transport", "tcp",
                                "-analyzeduration", "1000000",
                                "-probesize", "1000000",
                                "-fflags", "nobuffer+discardcorrupt",
                                "-flags", "low_delay",
                                "-i", rtsp_grab,
                                "-an",
                                "-frames:v", "1",
                                "-q:v", "3",
                                "-f", "image2",
                                "pipe:1",
                            ]
                            try:
                                grab_proc = await asyncio.create_subprocess_exec(
                                    *grab_cmd,
                                    stdout=asyncio.subprocess.PIPE,
                                    stderr=asyncio.subprocess.PIPE,
                                )
                                grab_out, _ = await asyncio.wait_for(grab_proc.communicate(), timeout=6)
                                if grab_out and len(grab_out) > 3_000:
                                    jpeg = grab_out
                            except asyncio.TimeoutError:
                                _LOGGER.warning("Detection cam%d (extra): RTSP grab timeout", ch)
                                try:
                                    grab_proc.kill()
                                except Exception:
                                    pass
                            except Exception as exc:
                                _LOGGER.debug("Detection cam%d (extra): RTSP grab fail: %s", ch, exc)
                        if not jpeg:
                            continue
                        snap_ts = now_m
                        _nvr_snap_cache[ch] = (now_m, jpeg)
                    else:
                        # NVR Dahua: HTTP snapshot endpoint с retry при disconnect
                        url  = f"http://{nvr_settings.host}/cgi-bin/snapshot.cgi?channel={ch}"
                        jpeg = b""
                        for attempt in range(2):
                            try:
                                resp = await client.get(url, auth=auth)
                                if resp.status_code == 200:
                                    jpeg = resp.content
                                    break
                            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError) as exc:
                                if attempt == 0:
                                    await asyncio.sleep(0.5)
                                    continue
                                _LOGGER.debug("Detection cam%d: NVR disconnect (attempt %d): %s", ch, attempt+1, exc)
                        if not jpeg:
                            continue

                        # Ако снимката е < 20 KB, NVR-ът е върнал "smart crop"
                        # (IVS/SMD изрязан кадър само около движещия се обект).
                        # Fallback: взимаме пълен кадър от RTSP sub-stream.
                        if len(jpeg) < 20_000:
                            _LOGGER.debug(
                                "Detection cam%d: HTTP snapshot е само %d B → RTSP fallback",
                                ch, len(jpeg),
                            )
                            rtsp_fb = (
                                f"rtsp://{nvr_settings.user}:{nvr_settings.password}@"
                                f"{nvr_settings.host}:554/cam/realmonitor"
                                f"?channel={ch}&subtype=1"
                            )
                            fb_cmd = [
                                "ffmpeg", "-y",
                                "-rtsp_transport", "tcp",
                                "-analyzeduration", "5000000",
                                "-probesize", "5000000",
                                "-i", rtsp_fb,
                                "-an",
                                "-vf", "select=eq(pict_type\\,I)",
                                "-vsync", "vfr",
                                "-frames:v", "1",
                                "-q:v", "3",
                                "-f", "image2",
                                "pipe:1",
                            ]
                            try:
                                fb_proc = await asyncio.create_subprocess_exec(
                                    *fb_cmd,
                                    stdout=asyncio.subprocess.PIPE,
                                    stderr=asyncio.subprocess.PIPE,
                                )
                                fb_out, _ = await asyncio.wait_for(
                                    fb_proc.communicate(), timeout=15
                                )
                                if fb_out and len(fb_out) > 5_000:
                                    jpeg = fb_out
                            except Exception as fb_exc:
                                _LOGGER.warning(
                                    "Detection cam%d: RTSP fallback грешка: %s", ch, fb_exc
                                )

                        snap_ts = now_m
                        _nvr_snap_cache[ch] = (now_m, jpeg)

                    # Пропускаме ако вече сме обработили точно този snapshot
                    if snap_ts == last_snap_ts.get(ch):
                        continue
                    last_snap_ts[ch] = snap_ts

                    # Per-camera confidence override или глобалната стойност
                    cam_conf = _detect_conf.get(ch)
                    eff_conf = cam_conf if cam_conf is not None else conf_thresh

                    def _infer(data: bytes = jpeg, _conf: float = eff_conf) -> tuple[list, list]:
                        img    = _PilImage.open(_bio.BytesIO(data)).convert("RGB")
                        img_np = _np.array(img)
                        # Per-class threshold: за хора по-нисък (рядко false positive)
                        person_conf = detect_settings.confidence_person
                        infer_conf  = min(_conf, person_conf)
                        results = model.predict(
                            img_np,
                            classes=detect_cls,
                            conf=infer_conf,
                            verbose=False,
                            save=False,
                        )
                        found = []
                        boxes = []
                        for r in results:
                            for box in r.boxes:
                                cls_id = int(box.cls[0])
                                conf_v = float(box.conf[0])
                                threshold = person_conf if cls_id == 0 else _conf
                                if conf_v < threshold:
                                    continue
                                lbl = _DETECT_LABELS.get(cls_id, str(cls_id))
                                found.append((lbl, round(conf_v, 3)))
                                xyxy = box.xyxy[0].tolist()
                                boxes.append({
                                    "label": lbl,
                                    "box": [round(v) for v in xyxy],
                                })
                        return found, boxes

                    detections, det_boxes = await loop.run_in_executor(None, _infer)

                    if not detections:
                        continue

                    # Object tracking: всеки засечен обект получава стабилен track_id.
                    # Event се записва САМО ако има поне един НОВ track (нов обект ИЛИ
                    # върнал се след > TRACK_TIMEOUT_S).
                    now      = time.monotonic()
                    tracked  = _track_update(ch, det_boxes, now)
                    new_objs = [t for t in tracked if t["is_new"]]
                    if not new_objs:
                        continue

                    # Throttle: минимум N секунди между 2 event-а от 1 камера
                    if now - last_event.get(ch, 0) < detect_settings.cooldown:
                        continue
                    last_event[ch] = now

                    detected_at   = datetime.now(timezone.utc)
                    # В labels записваме САМО новите обекти (за обогатяване с face names)
                    labels        = [t["label"] for t in new_objs]
                    # confs няма пряк mapping към new_objs → пазим всички confidences (для статистика)
                    confs         = [d[1] for d in detections]
                    unique_labels = list(dict.fromkeys(labels))

                    cam_dir  = EVENTS_DIR / str(ch)
                    cam_dir.mkdir(parents=True, exist_ok=True)
                    ts_str   = detected_at.strftime("%Y%m%d_%H%M%S")
                    img_path = cam_dir / f"{ts_str}_{'_'.join(unique_labels)}.jpg"

                    # При НVR_DETECT_HQ=true вземаме HQ кадър от main stream за запис.
                    # YOLO inference върви на ниско-резолюционния кеш snapshot;
                    # само при потвърдена детекция правим допълнителна RTSP заявка.
                    save_jpeg = jpeg
                    if detect_settings.hq and nvr_settings.host:
                        _hq_rtsp = (
                            f"rtsp://{nvr_settings.user}:{nvr_settings.password}@"
                            f"{nvr_settings.host}:554/cam/realmonitor"
                            f"?channel={ch}&subtype=0"
                        )
                        _hq_cmd = [
                            "ffmpeg", "-y",
                            "-rtsp_transport", "tcp",
                            "-analyzeduration", "5000000",
                            "-probesize", "5000000",
                            "-i", _hq_rtsp,
                            "-an",
                            "-vf", "select=eq(pict_type\\,I)",
                            "-vsync", "vfr",
                            "-frames:v", "1",
                            "-q:v", "2",
                            "-f", "image2",
                            "pipe:1",
                        ]
                        try:
                            _hq_proc = await asyncio.create_subprocess_exec(
                                *_hq_cmd,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE,
                            )
                            _hq_out, _ = await asyncio.wait_for(
                                _hq_proc.communicate(), timeout=12
                            )
                            if _hq_out and len(_hq_out) > 10_000:
                                save_jpeg = _hq_out
                        except Exception as hq_exc:
                            _LOGGER.debug("Detection cam%d: HQ snapshot грешка: %s", ch, hq_exc)

                    await asyncio.to_thread(img_path.write_bytes, save_jpeg)
                    # tracks: само новите обекти (новопоявили се или върнали се след > TRACK_TIMEOUT_S)
                    tracks_meta = [
                        {"track_id": t["track_id"], "label": t["label"], "box": t["box"]}
                        for t in new_objs
                    ]
                    await asyncio.to_thread(
                        _db_insert_detection,
                        ch, detected_at.isoformat(), labels, confs, str(img_path),
                        det_boxes, tracks_meta,
                    )
                    await asyncio.to_thread(_db_trim_events, 1000)
                    _LOGGER.info(
                        "🎯 Детекция cam%d: NEW %s (tracks %s) [%s]",
                        ch, labels, [t["track_id"] for t in new_objs],
                        "HQ" if save_jpeg is not jpeg else "SD",
                    )

                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    _LOGGER.warning("Detection cam%d: %s", ch, exc)

            await asyncio.sleep(detect_settings.interval)


# ── 24/7 Continuous Recording ──────────────────────────────────────────────
# Per camera FFmpeg subprocess pulls RTSP main stream и сегментира в 1-min MP4
# на /recordings/{ch}/YYYYMMDD/HH/MM.mp4. Watchdog проверява изхода и restartва
# процеса при срив (мрежа, реboot на камерата).

_recording_procs:    dict[int, asyncio.subprocess.Process] = {}
_recording_stats:    dict[int, dict] = {}  # ch → {started_at, restarts, last_segment_ts}
_recording_armed:    dict[int, bool] = {}  # ch → armed (default True; loaded from meta at start)
_recording_ch_names: dict[int, str]  = {}  # ch → friendly name (за restart)


def _last_segment_mtime(ch: int) -> float | None:
    """Връща mtime на най-новия .mp4 segment за камерата (или None)."""
    cam_dir = _ch_recording_dir(ch)
    if not cam_dir.exists():
        return None
    latest_mtime = 0.0
    try:
        # Само истинските day дирове (YYYYMMDD), без "live" и др.
        day_dirs = [
            d for d in sorted(cam_dir.iterdir(), reverse=True)
            if d.is_dir() and d.name != "live" and d.name.isdigit()
        ]
        for day_dir in day_dirs[:2]:
            # ВАЖНО: recorder start-ът пре-създава ВСИЧКИ 24 часови дирове за
            # деня (мн. от тях празни/бъдещи). Затова НЕ ограничаваме до top-2
            # часа — обхождаме всички в обратен ред докато намерим час със
            # segment-и. Иначе празните часове "23","22" → latest=0 → None →
            # watchdog фалшиво рестартира recorder-а (regression: причиняваше
            # WHEP churn → flip към MJPEG за Shelly камерите).
            for hour_dir in sorted(day_dir.iterdir(), reverse=True):
                if not hour_dir.is_dir():
                    continue
                for mp4 in hour_dir.glob("*.mp4"):
                    try:
                        m = mp4.stat().st_mtime
                        if m > latest_mtime:
                            latest_mtime = m
                    except Exception:
                        pass
                if latest_mtime > 0:
                    break  # намерихме най-новия час със segment-и за този ден
            if latest_mtime > 0:
                break
    except Exception:
        return None
    return latest_mtime if latest_mtime > 0 else None


def _rtsp_url_for_ch(ch: int) -> str | None:
    """RTSP URL за канал. Подкрепя:
      • RTSP/Reolink камери (type='rtsp_generic') — собствен URL в config
      • NVR Dahua канали (type='nvr_dahua') — host/user/password в config
    """
    cam = _get_camera(ch)
    if not cam:
        return None
    cfg = cam["config"]

    if cam["type"] == "rtsp_generic":
        url_main = cfg.get("rtsp_main", "")
        url_sub  = cfg.get("rtsp_sub", "")
        return url_main if recording_settings.quality == "main" else (url_sub or url_main)

    if cam["type"] == "nvr_dahua":
        host = cfg.get("host") or nvr_settings.host
        user = cfg.get("user") or nvr_settings.user
        pwd  = cfg.get("password") or nvr_settings.password
        if not (host and user and pwd):
            return None
        ch_id = cfg.get("channel_id", ch)
        subtype = 0 if recording_settings.quality == "main" else 1
        u = quote(user, safe="")
        p = quote(pwd, safe="")
        return (
            f"rtsp://{u}:{p}@{host}:554/cam/realmonitor"
            f"?channel={ch_id}&subtype={subtype}"
        )

    return None


def _ch_recording_dir(ch: int) -> Path:
    return Path(recording_settings.path) / str(ch)


def _hls_latest_ts(ch: int) -> Path | None:
    """Връща пътя до НАЙ-СКОРОШНИЯ HLS .ts segment за канала, или None
    ако recording не е активно или не е писал нищо.

    Позволява detection loop да grab-не frame от вече-recorded segment
    вместо да отваря нов RTSP/HTTP connection към камерата (16× по-бързо).
    """
    live_dir = _ch_recording_dir(ch) / "live"
    if not live_dir.is_dir():
        return None
    try:
        ts_files = [f for f in live_dir.iterdir() if f.suffix == ".ts"]
        if not ts_files:
            return None
        # Сегмент който е валиден (не е току-що отворен с 0 байта)
        # вземаме предпоследния (последният може все още да се пише)
        ts_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        # Прескачаме файлове с age < 0.3s (могат да са непълни)
        now = time.time()
        for f in ts_files:
            if (now - f.stat().st_mtime) > 0.3 and f.stat().st_size > 10_000:
                return f
        return ts_files[0] if ts_files else None
    except Exception:
        return None


# Shelly WebRTC grab процеси (per-channel) — shelly-webrtc-grab binary → ffmpeg stdin pipe
_recording_feeders: dict[int, asyncio.Task]                    = {}  # остарял (JPEG poll) — запазен за съвместимост
_shelly_webrtc_procs: dict[int, asyncio.subprocess.Process]    = {}  # Go WebRTC subprocess per channel

SHELLY_RECORDING_FPS = 5      # output framerate
SHELLY_RECORDING_RES = "1280x720"  # реална резолюция на /camera/0/snapshot от новия firmware

# RTSP порт на вградения Shelly RTSP сървър (malmStreamer). Firmware ≥ есен 2026
# (Shelly/fw/shelly-ng → libs/shelly-camera): `camera.rtsp.enable`, default false.
SHELLY_RTSP_PORT = 554


def _shelly_rtsp_url(ip: str, stream: int = 0) -> str:
    """RTSP URL на вградения Shelly RTSP сървър.

    stream 0 = main (H.264 1920×1080 @ 25fps + AAC), stream 1 = secondary
    (640×360 @ 10fps). Формат: `rtsp://<ip>:554/stream/<idx>`.
    """
    return f"rtsp://{ip}:{SHELLY_RTSP_PORT}/stream/{stream}"


async def _shelly_ensure_rtsp(ip: str, timeout_s: float = 6.0) -> bool:
    """Проверява за вграден RTSP сървър и го включва ако трябва.

    `Camera.GetConfig` → ако липсва `rtsp` ключ → стар firmware без RTSP → False.
    Ако `rtsp.enable` е false → `Camera.SetConfig {rtsp:{enable:true}}` (без
    restart на новия firmware). Връща True само ако RTSP е включен → recorder-ът
    минава по унифицирания RTSP `-c copy` път (както NVR). False → fallback WebRTC.
    """
    base = f"http://{ip}"
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.get(f"{base}/rpc/Camera.GetConfig", params={"id": 0})
            if r.status_code != 200:
                return False
            rtsp = r.json().get("rtsp")
            if not isinstance(rtsp, dict):
                return False  # firmware без RTSP поддръжка
            if rtsp.get("enable") is True:
                return True
            sr = await client.post(
                f"{base}/rpc/Camera.SetConfig",
                json={"id": 0, "config": {"rtsp": {"enable": True}}},
            )
            if sr.status_code != 200:
                return False
            r2 = await client.get(f"{base}/rpc/Camera.GetConfig", params={"id": 0})
            return bool(r2.status_code == 200 and (r2.json().get("rtsp") or {}).get("enable"))
    except Exception:
        return False


async def _shelly_jpeg_feeder(
    ch: int,
    ip: str,
    proc: asyncio.subprocess.Process,
    fps: int = SHELLY_RECORDING_FPS,
) -> None:
    """Polls Shelly /camera/0/snapshot @ fps Hz и пуска JPEG-и в ffmpeg stdin.

    Завършва при stdin BrokenPipeError (ffmpeg е exit-нал).
    """
    interval = 1.0 / fps
    url = f"http://{ip}/camera/0/snapshot"
    next_t = time.monotonic()
    consecutive_errors = 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as client:
        while proc.returncode is None:
            try:
                resp = await client.get(url)
                if resp.status_code == 200 and resp.content:
                    proc.stdin.write(resp.content)
                    await proc.stdin.drain()
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1
            except (BrokenPipeError, ConnectionResetError):
                _LOGGER.debug("Shelly cam%d feeder: stdin closed", ch)
                return
            except Exception as exc:
                consecutive_errors += 1
                if consecutive_errors == 1 or consecutive_errors % 50 == 0:
                    _LOGGER.warning(
                        "Shelly cam%d feeder: %s (consecutive=%d)",
                        ch, repr(exc) if not str(exc) else str(exc), consecutive_errors,
                    )
                if consecutive_errors > 200:
                    _LOGGER.warning("Shelly cam%d feeder: too many errors → exit", ch)
                    return
            next_t += interval
            sleep_for = max(0.0, next_t - time.monotonic())
            if sleep_for == 0.0:
                # zaostane sme — reset baseline за да не "натрупваме"
                next_t = time.monotonic()
            else:
                await asyncio.sleep(sleep_for)


async def _spawn_ffmpeg_recorder(ch: int, ch_name: str) -> asyncio.subprocess.Process | None:
    """Стартира FFmpeg subprocess със 2 output-а:
      1. 1-мин MP4 segments (за timeline / архив)
      2. HLS rolling window (за live viewing — споделяме един input stream)

    Поддържа 2 типа източници:
      • NVR RTSP (Dahua HEVC) → `-c copy` (нула CPU, без transcode)
      • Shelly polled JPEG    → `libx264 -preset ultrafast` (~3-5% CPU/камера)
    """
    cam_dir = _ch_recording_dir(ch)
    cam_dir.mkdir(parents=True, exist_ok=True)

    # Live HLS директория — изчиства се при start
    hls_dir = cam_dir / "live"
    if hls_dir.exists():
        for f in hls_dir.iterdir():
            try:
                f.unlink()
            except Exception:
                pass
    hls_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = str(cam_dir / "%Y%m%d" / "%H" / "%M.mp4")
    hls_playlist   = str(hls_dir / "live.m3u8")
    hls_seg_pat    = str(hls_dir / "live%d.ts")

    shelly_cam = next((c for c in _shelly_cams_list() if c["ch"] == ch), None)
    is_shelly = shelly_cam is not None
    shelly_mode = None  # 'rtsp' | 'webrtc'

    if is_shelly:
        # Auto-detect: предпочитаме вградения Shelly RTSP сървър (firmware ≥ есен
        # 2026) — H.264 + AAC с реални RTP timestamps → `-c copy`, 0 CPU,
        # унифициран с NVR пътя (без pion/PTS проблемите). Fallback → WebRTC
        # (shelly-webrtc-grab / WHEP) за стар firmware без RTSP.
        env_mode = (os.getenv("SHELLY_RECORDING_MODE") or "auto").strip().lower()
        if env_mode in ("rtsp", "webrtc"):
            shelly_mode = env_mode
        elif await _shelly_ensure_rtsp(shelly_cam["ip"]):
            shelly_mode = "rtsp"
        else:
            shelly_mode = "webrtc"
        _LOGGER.info("Recording cam%d: Shelly режим=%s", ch, shelly_mode)

    if is_shelly and shelly_mode == "rtsp":
        # Вграден Shelly RTSP сървър → същият път като NVR (`-c copy`).
        stream_idx = 0 if recording_settings.quality == "main" else 1
        input_args = [
            "-rtsp_transport", "tcp",
            "-timeout", "30000000",
            "-i", _shelly_rtsp_url(shelly_cam["ip"], stream_idx),
        ]
        audio_args = [] if recording_settings.audio else ["-an"]
        codec_args = [
            "-map", "0:v",
            *(["-map", "0:a?"] if recording_settings.audio else []),
            "-c", "copy",
            *audio_args,
        ]
        seg_codec  = codec_args
        hls_codec  = list(codec_args)
        stdin_arg  = asyncio.subprocess.DEVNULL
    elif is_shelly:
        # WebRTC fallback: shelly-webrtc-grab → OS pipe → ffmpeg stdin (H.264
        # Annex-B). Суровият H.264 НЯМА timestamps → `-use_wallclock_as_timestamps`
        # дава монотонни PTS (иначе HLS muxer-ът → rc=183 restart loop).
        input_args = [
            "-use_wallclock_as_timestamps", "1",
            "-fflags", "+genpts",
            "-analyzeduration", "10M",
            "-probesize", "10M",
            "-f", "h264",
            "-i", "pipe:0",   # read H.264 Annex-B from stdin (pipe read-end)
        ]
        codec_args = [
            "-c:v", "copy",   # нула CPU — директно копиране на H.264
            "-an",
        ]
        seg_codec  = codec_args
        hls_codec  = codec_args
        stdin_arg  = None     # ще се зададе с os.pipe() fd по-долу
    else:
        rtsp = _rtsp_url_for_ch(ch)
        if not rtsp:
            _LOGGER.debug("Recording cam%d: no RTSP URL", ch)
            return None
        input_args = [
            "-rtsp_transport", "tcp",
            "-timeout", "30000000",
            "-i", rtsp,
        ]
        audio_args = [] if recording_settings.audio else ["-an"]
        seg_codec = [
            "-map", "0:v",
            *(["-map", "0:a?"] if recording_settings.audio else []),
            "-c", "copy",
            *audio_args,
        ]
        hls_codec = list(seg_codec)
        stdin_arg = asyncio.subprocess.DEVNULL

    cmd = [
        "ffmpeg",
        "-loglevel", "warning",
        *input_args,
        # ── Output 1: 1-минутни MP4 segments ──
        *seg_codec,
        "-f", "segment",
        "-segment_time", str(recording_settings.segment_s),
        "-segment_atclocktime", "1",
        "-reset_timestamps", "1",
        "-strftime", "1",
        "-segment_format_options", "movflags=+faststart",
        output_pattern,
        # ── Output 2: HLS rolling live (~6 сек buffer) ──
        *hls_codec,
        "-f", "hls",
        "-hls_time", "2",
        "-hls_list_size", "3",
        "-hls_flags", "delete_segments+omit_endlist+independent_segments",
        "-hls_segment_type", "mpegts",
        "-hls_segment_filename", hls_seg_pat,
        hls_playlist,
    ]

    if is_shelly:
        src_type = "Shelly RTSP (H.264 + AAC copy)" if shelly_mode == "rtsp" else "Shelly WebRTC (H.264 1920×1080)"
    else:
        src_type = "NVR RTSP"
    _LOGGER.info(
        "Recording cam%d (%s, %s): стартирам FFmpeg → %s + live HLS",
        ch, ch_name, src_type, cam_dir,
    )
    try:
        now = datetime.now()
        for h in range(24):
            (cam_dir / now.strftime("%Y%m%d") / f"{h:02d}").mkdir(parents=True, exist_ok=True)

        if is_shelly and shelly_mode == "webrtc":
            # OS pipe: Go binary stdout → ffmpeg stdin
            import os as _os
            r_fd, w_fd = _os.pipe()

            # Спираме предишен WebRTC процес ако съществува
            old_wrtc = _shelly_webrtc_procs.pop(ch, None)
            if old_wrtc and old_wrtc.returncode is None:
                try:
                    old_wrtc.terminate()
                except Exception:
                    pass

            cam_url = f"http://{shelly_cam['ip']}"
            go_proc = await asyncio.create_subprocess_exec(
                "shelly-webrtc-grab", cam_url,
                stdout=w_fd,
                stderr=asyncio.subprocess.PIPE,
            )
            _os.close(w_fd)  # parent не се нуждае от write-end
            _shelly_webrtc_procs[ch] = go_proc

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=r_fd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _os.close(r_fd)  # parent не се нуждае от read-end след предаването

        else:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=stdin_arg,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )

        return proc
    except Exception as exc:
        _LOGGER.warning("Recording cam%d: spawn fail: %s", ch, exc)
        return None


async def _ensure_subdirs_loop(channels: list[int]) -> None:
    """Background task: предварително създава поддиректории за следващия час."""
    while True:
        try:
            now = datetime.now()
            day_str = now.strftime("%Y%m%d")
            for ch in channels:
                cam_dir = _ch_recording_dir(ch)
                # Текущ + следващ час
                for h_offset in (0, 1):
                    h = (now.hour + h_offset) % 24
                    (cam_dir / day_str / f"{h:02d}").mkdir(parents=True, exist_ok=True)
                # При смяна на ден предварително следващия
                if (now.hour + 1) >= 24:
                    next_day = (now + timedelta(days=1)).strftime("%Y%m%d")
                    (cam_dir / next_day / "00").mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            _LOGGER.debug("subdir create: %s", exc)
        await asyncio.sleep(300)  # на всеки 5 мин


async def _stop_recorder(ch: int) -> None:
    """Чисто терминирай FFmpeg recorder + Shelly WebRTC grab (ако има) за камера."""
    feeder = _recording_feeders.pop(ch, None)
    if feeder and not feeder.done():
        feeder.cancel()
    wrtc = _shelly_webrtc_procs.pop(ch, None)
    if wrtc and wrtc.returncode is None:
        try:
            wrtc.terminate()
            await asyncio.wait_for(wrtc.wait(), 3.0)
        except Exception:
            try:
                wrtc.kill()
            except Exception:
                pass
    proc = _recording_procs.pop(ch, None)
    if not proc or proc.returncode is not None:
        return
    try:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), 5.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
        _LOGGER.info("Recording cam%d: спрян", ch)
    except Exception as exc:
        _LOGGER.debug("stop recorder cam%d: %s", ch, exc)


async def _start_recorder(ch: int) -> bool:
    """Стартира recorder за камера (ако вече не работи)."""
    if ch in _recording_procs and _recording_procs[ch].returncode is None:
        return True
    name = _recording_ch_names.get(ch, f"Камера {ch}")
    proc = await _spawn_ffmpeg_recorder(ch, name)
    if proc:
        _recording_procs[ch] = proc
        st = _recording_stats.setdefault(ch, {
            "started_at":      datetime.now(timezone.utc).isoformat(),
            "restarts":        0,
            "last_segment_ts": None,
        })
        st["started_at"] = datetime.now(timezone.utc).isoformat()
        return True
    return False


async def _recording_loop() -> None:
    """Главен loop: spawn-ва FFmpeg recorders за armed камери + watchdog за рестарт."""
    if not recording_settings.enabled:
        _LOGGER.info("Continuous recording: disabled")
        return

    Path(recording_settings.path).mkdir(parents=True, exist_ok=True)

    # Списъкът камери се чете от DB cache (всички enabled, всеки тип).
    # При CRUD промени cache-а се invalidate-ва → recording_loop забелязва
    # промените на следващия watchdog cycle (виж по-долу).
    def _build_channel_list() -> list[int]:
        out: list[int] = []
        for cam in _cameras_get_all().values():
            if not cam["enabled"]:
                continue
            ch = cam["channel"]
            out.append(ch)
            _recording_ch_names[ch] = cam["name"]
        return sorted(out)

    channels = _build_channel_list()
    if not channels:
        _LOGGER.info("Continuous recording: no channels configured")
        return

    # Зареждаме per-camera armed state от SQLite meta (default = armed)
    for ch in channels:
        try:
            stored = await asyncio.to_thread(_db_meta_get, f"recording_armed_{ch}", "1")
            _recording_armed[ch] = (stored == "1")
        except Exception:
            _recording_armed[ch] = True

    # Background helper: създава поддиректории за strftime
    asyncio.create_task(_ensure_subdirs_loop(channels))

    # Spawn-ваме всички armed recorder-и
    for ch in channels:
        if _recording_armed.get(ch, True):
            await _start_recorder(ch)
            await asyncio.sleep(0.5)  # stagger

    _LOGGER.info(
        "Continuous recording: %d/%d камери armed и стартирани",
        len([p for p in _recording_procs.values() if p.returncode is None]),
        len(channels),
    )

    # Watchdog: проверява процеси + mtime на последен segment на всеки 30 сек
    # Ако процес е zombie (живи, но не пише segments > 120 сек) → kill и restart
    STALL_THRESHOLD_S = 120.0
    while True:
        await asyncio.sleep(30)
        try:
            for ch in list(channels):
                if not _recording_armed.get(ch, True):
                    if ch in _recording_procs and _recording_procs[ch].returncode is None:
                        await _stop_recorder(ch)
                    continue
                proc = _recording_procs.get(ch)
                need_restart = False
                reason = ""
                if proc is None or proc.returncode is not None:
                    need_restart = True
                    reason = f"процесът мъртъв (rc={proc.returncode if proc else 'missing'})"
                else:
                    # Проверка по mtime: ако последен segment е > STALL_THRESHOLD_S → zombie
                    last_mtime = _last_segment_mtime(ch)
                    if last_mtime is not None:
                        age = time.time() - last_mtime
                        if age > STALL_THRESHOLD_S:
                            need_restart = True
                            reason = f"zombie (последен segment преди {age:.0f}s)"
                    else:
                        # Никакъв segment засега — проверка спрямо started_at
                        st = _recording_stats.get(ch, {})
                        try:
                            sa = datetime.fromisoformat(st.get("started_at",""))
                            age = (datetime.now(timezone.utc) - sa).total_seconds()
                            if age > STALL_THRESHOLD_S:
                                need_restart = True
                                reason = f"никакъв segment {age:.0f}s след старт"
                        except Exception:
                            pass

                if need_restart:
                    _LOGGER.warning("Recording cam%d: %s → restart", ch, reason)
                    if proc and proc.stderr:
                        try:
                            tail = await asyncio.wait_for(proc.stderr.read(2000), 1.0)
                            if tail:
                                _LOGGER.debug("ffmpeg stderr cam%d: %s", ch, tail.decode(errors="ignore")[-500:])
                        except Exception:
                            pass
                    # Пълно спиране — вкл. Shelly Go WebRTC процеса + feeder-а.
                    # Иначе старият shelly-webrtc-grab остава жив, държи WHEP
                    # сесията на камерата → новият handshake получава 5xx →
                    # recorder-ът фалшиво минава на MJPEG fallback.
                    await _stop_recorder(ch)
                    if await _start_recorder(ch):
                        st = _recording_stats.setdefault(ch, {})
                        st["restarts"] = st.get("restarts", 0) + 1
                        st["last_restart"] = datetime.now(timezone.utc).isoformat()

                # Update last_segment_ts (за UI)
                m = _last_segment_mtime(ch)
                if m is not None:
                    _recording_stats.setdefault(ch, {})["last_segment_ts"] = (
                        datetime.fromtimestamp(m, timezone.utc).isoformat()
                    )
        except Exception as exc:
            _LOGGER.warning("Recording watchdog: %s", exc)


async def _recording_cleanup_loop() -> None:
    """Изтрива дни > retention_days. При запълване над disk_limit_pct → намалява retention.

    Стратегия: изтриваме най-стари дни за всяка камера докато:
    - дните > retention_days  — обикновен cleanup
    - disk usage > disk_limit_pct  — намаляваме retention с 1 ден докато стане OK
    """
    if not recording_settings.enabled:
        return

    base = Path(recording_settings.path)
    base.mkdir(parents=True, exist_ok=True)
    await asyncio.sleep(120)  # wait для recorder loop

    while True:
        try:
            now       = datetime.now()
            keep_days = recording_settings.retention_days   # global default

            # Per-camera retention map: {channel: keep_days}.
            # Камера без override → ползва глобалния keep_days.
            cams_by_channel = {c["channel"]: c for c in _cameras_get_all().values()}
            def _keep_for_dir(cam_dir_name: str) -> int:
                try:
                    ch = int(cam_dir_name)
                except ValueError:
                    return keep_days
                cam = cams_by_channel.get(ch)
                if cam and cam.get("retention_days"):
                    return int(cam["retention_days"])
                return keep_days

            # 1) Обикновен retention cleanup (per-camera retention)
            for cam_dir in base.iterdir():
                if not cam_dir.is_dir():
                    continue
                cam_keep = _keep_for_dir(cam_dir.name)
                cutoff = (now - timedelta(days=cam_keep)).strftime("%Y%m%d")
                for day_dir in cam_dir.iterdir():
                    if not day_dir.is_dir():
                        continue
                    if day_dir.name < cutoff:
                        try:
                            for h_dir in day_dir.iterdir():
                                for f in h_dir.iterdir():
                                    f.unlink(missing_ok=True)
                                h_dir.rmdir()
                            day_dir.rmdir()
                            _LOGGER.info("Recording cleanup: изтрит %s (keep=%dd)",
                                         day_dir, cam_keep)
                        except Exception as exc:
                            _LOGGER.debug("cleanup %s: %s", day_dir, exc)

            # 2) Disk-full fallback: ако > disk_limit_pct, намалявай retention
            try:
                stat = shutil.disk_usage(base)
                used_pct = (stat.used / stat.total) * 100.0
                while used_pct > recording_settings.disk_limit_pct and keep_days > 1:
                    keep_days -= 1
                    new_cutoff = (now - timedelta(days=keep_days)).strftime("%Y%m%d")
                    deleted_any = False
                    for cam_dir in base.iterdir():
                        if not cam_dir.is_dir():
                            continue
                        for day_dir in cam_dir.iterdir():
                            if not day_dir.is_dir():
                                continue
                            if day_dir.name < new_cutoff:
                                try:
                                    for h_dir in day_dir.iterdir():
                                        for f in h_dir.iterdir():
                                            f.unlink(missing_ok=True)
                                        h_dir.rmdir()
                                    day_dir.rmdir()
                                    deleted_any = True
                                except Exception:
                                    pass
                    _LOGGER.warning(
                        "Recording disk-full: %.1f%% used → намалявам retention на %d дни",
                        used_pct, keep_days,
                    )
                    if not deleted_any:
                        break
                    stat = shutil.disk_usage(base)
                    used_pct = (stat.used / stat.total) * 100.0
            except Exception as exc:
                _LOGGER.debug("disk-full fallback: %s", exc)

        except Exception as exc:
            _LOGGER.warning("Recording cleanup: %s", exc)

        await asyncio.sleep(3600)  # на всеки час


# ── Shelly background detection loop ───────────────────────────────────────
async def _shelly_detection_loop() -> None:
    """Периодично взима 640×360 JPEG от всяка Shelly камера и пуска YOLO."""
    if not detect_settings.enabled:
        return

    loop = asyncio.get_running_loop()
    try:
        import io as _bio
        import numpy as _np
        from PIL import Image as _PilImage
        from ultralytics import YOLO as _YOLO
    except ImportError:
        _LOGGER.error("ultralytics/pillow не са инсталирани — Shelly detection изключен")
        return

    _LOGGER.info("Shelly Detection: зарежда %s…", detect_settings.model)
    model = await loop.run_in_executor(None, lambda: _YOLO(detect_settings.model))
    _LOGGER.info("Shelly Detection: модел зареден ✓ (%s)", detect_settings.model)

    detect_cls  = [int(x.strip()) for x in detect_settings.classes.split(",") if x.strip()]
    conf_thresh = detect_settings.confidence
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)

    last_event: dict[int, float] = {}

    await asyncio.sleep(8)   # stagger спрямо NVR loop

    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        while True:
            for cam in _shelly_cams_list():
                ch, ip, cname = cam["ch"], cam["ip"], cam["name"]

                if not _detect_armed.get(ch, True):
                    continue

                try:
                    # Предпочитаме HLS .ts кадър (от WebRTC запис, 1920×1080)
                    # ако WebRTC recording вече работи и е генерирал .ts файл.
                    jpeg_sd = b""
                    ts_path = _hls_latest_ts(ch)
                    if ts_path is not None:
                        grab_cmd = [
                            "ffmpeg", "-y",
                            "-i", str(ts_path),
                            "-an", "-vframes", "1",
                            "-q:v", "3", "-update", "1",
                            "-f", "image2", "pipe:1",
                        ]
                        try:
                            gp = await asyncio.create_subprocess_exec(
                                *grab_cmd,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.DEVNULL,
                            )
                            raw, _ = await asyncio.wait_for(gp.communicate(), timeout=4)
                            if raw and len(raw) > 5_000:
                                jpeg_sd = raw
                        except Exception:
                            pass

                    if not jpeg_sd:
                        # Fallback: HTTP snapshot от камерата (1280×720)
                        resp = await client.get(f"http://{ip}/camera/0/snapshot")
                        if resp.status_code != 200 or len(resp.content) < 20_000:
                            continue
                        jpeg_sd = resp.content

                    def _infer(data: bytes, mdl=model) -> tuple[list, list]:
                        import io as _io, numpy as _np
                        from PIL import Image as _Pil
                        img_np = _np.array(_Pil.open(_io.BytesIO(data)).convert("RGB"))
                        person_conf = detect_settings.confidence_person
                        infer_conf  = min(conf_thresh, person_conf)
                        found, boxes = [], []
                        for r in mdl.predict(img_np, classes=detect_cls, conf=infer_conf,
                                             verbose=False, save=False):
                            for box in r.boxes:
                                cls_id = int(box.cls[0])
                                conf_v = float(box.conf[0])
                                threshold = person_conf if cls_id == 0 else conf_thresh
                                if conf_v < threshold:
                                    continue
                                lbl = _DETECT_LABELS.get(cls_id, str(cls_id))
                                found.append((lbl, round(conf_v, 3)))
                                xyxy = box.xyxy[0].tolist()
                                boxes.append({"label": lbl, "box": [round(v) for v in xyxy]})
                        return found, boxes

                    detections, det_boxes = await loop.run_in_executor(None, _infer, jpeg_sd)

                    if not detections:
                        continue

                    # Object tracking: event се записва САМО при нов track
                    now      = time.monotonic()
                    tracked  = _track_update(ch, det_boxes, now)
                    new_objs = [t for t in tracked if t["is_new"]]
                    if not new_objs:
                        continue

                    if now - last_event.get(ch, 0) < detect_settings.cooldown:
                        continue
                    last_event[ch] = now

                    detected_at   = datetime.now(timezone.utc)
                    labels        = [t["label"] for t in new_objs]
                    confs         = [d[1] for d in detections]
                    unique_labels = list(dict.fromkeys(labels))

                    # /camera/0/snapshot вече връща 1280×720 — директно ползваме
                    save_jpeg = jpeg_sd

                    cam_dir  = EVENTS_DIR / str(ch)
                    cam_dir.mkdir(parents=True, exist_ok=True)
                    ts_str   = detected_at.strftime("%Y%m%d_%H%M%S")
                    img_path = cam_dir / f"{ts_str}_{'_'.join(unique_labels)}.jpg"
                    img_path.write_bytes(save_jpeg)

                    tracks_meta = [
                        {"track_id": t["track_id"], "label": t["label"], "box": t["box"]}
                        for t in new_objs
                    ]
                    await asyncio.to_thread(
                        _db_insert_detection,
                        ch, detected_at.isoformat(), labels, confs, str(img_path),
                        det_boxes, tracks_meta,
                    )
                    await asyncio.to_thread(_db_trim_events, 1000)
                    _LOGGER.info(
                        "🎯 Shelly det %s ch%d: NEW %s (tracks %s) [%dKB]",
                        cname, ch, unique_labels, [t["track_id"] for t in new_objs],
                        len(save_jpeg) // 1024,
                    )

                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    _LOGGER.warning("Shelly detection ch%d: %s", ch, exc)

            await asyncio.sleep(detect_settings.interval)


# ── Face migration: dlib 128-d → InsightFace ArcFace 512-d ─────────────────
def _migrate_one_crop_to_v2(crop_id: int, image_path: str, app) -> bool:
    """Пресмята v2 embedding за съществуващ face crop файл и го записва.

    Старите dlib crop файлове са малки (~80-200px) и съдържат само лицето.
    SCRFD очаква по-широка сцена с лицето като ~5-30% от кадъра, затова
    тук pad-ваме crop-а до 640×640 с лицето заемащо ~50% от centre, на
    черен фон. Това позволява SCRFD да го детектне надеждно.

    Връща True ако успешно, False ако не може (липсващ файл, no face detected).
    """
    import cv2
    import numpy as _np

    p = Path(image_path)
    if not p.exists():
        return False
    img_bgr = cv2.imread(str(p))
    if img_bgr is None:
        return False

    # Pad до 640×640 ако crop-а е малък — SCRFD работи по-добре на нормална сцена
    h, w = img_bgr.shape[:2]
    if h < 480 or w < 480:
        target = 640
        # Scale до ~50% от target, така че да има context margin около лицето
        scale = (target * 0.5) / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        canvas = _np.zeros((target, target, 3), dtype=_np.uint8)
        oy = (target - new_h) // 2
        ox = (target - new_w) // 2
        canvas[oy:oy + new_h, ox:ox + new_w] = resized
        img_bgr = canvas

    try:
        faces = app.get(img_bgr)
    except Exception as exc:
        _LOGGER.debug("Migrate crop #%d: InsightFace err: %s", crop_id, exc)
        return False
    if not faces:
        return False
    # Crop файлът обикновено съдържа 1 лице — взимаме най-голямото за надеждност
    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    # По-нисък threshold защото знаем че crop-ът съдържа лице (вече е била детекция от dlib)
    if best.det_score < 0.3:
        return False
    _db_set_crop_embedding_v2(crop_id, best.embedding.tolist())
    return True


async def _migrate_faces_to_v2() -> None:
    """One-time миграция: пресмята InsightFace ArcFace 512-d embeddings за всички
    съществуващи face crop файлове. Безопасно за повторно изпълнение
    (само crops без v2 embedding се обработват)."""
    if _db_meta_get("face_v2_migrated", "0") == "1":
        return  # вече мигрирано

    app = await asyncio.to_thread(_get_insightface_app)
    if app is None:
        _LOGGER.warning("Face v2 migration: InsightFace недостъпен — skip")
        return

    crops = await asyncio.to_thread(_db_get_crops_without_v2)
    if not crops:
        await asyncio.to_thread(_db_meta_set, "face_v2_migrated", "1")
        _LOGGER.info("Face v2 migration: 0 crops за обработка — DONE ✓")
        return

    _LOGGER.info(
        "Face v2 migration: започвам re-encode на %d crops към ArcFace 512-d ...",
        len(crops),
    )
    loop = asyncio.get_running_loop()
    success = fail = 0
    t0 = time.monotonic()
    for crop_id, image_path, cluster_id in crops:
        try:
            ok = await loop.run_in_executor(
                None, _migrate_one_crop_to_v2, crop_id, image_path, app,
            )
            if ok: success += 1
            else: fail += 1
        except Exception as exc:
            _LOGGER.warning("Face v2 migration crop #%d: %s", crop_id, exc)
            fail += 1
    _LOGGER.info(
        "Face v2 migration: %d/%d успешни, %d пропуснати (%.1fs) ✓",
        success, len(crops), fail, time.monotonic() - t0,
    )
    await asyncio.to_thread(_db_meta_set, "face_v2_migrated", "1")


# ── Face extraction background loop ────────────────────────────────────────
async def _face_extraction_loop() -> None:
    """Background: извлича лица от detection_events с 'person', клъстерира ги."""
    _LOGGER.info("👤 Face loop: ENTRY (sleeping 45s)")
    try:
        await asyncio.sleep(45)
    except Exception as exc:
        _LOGGER.error("👤 Face loop CRASH at sleep: %s", exc, exc_info=True)
        return

    _LOGGER.info("👤 Face loop: woke up, loading InsightFace ...")
    loop = asyncio.get_running_loop()

    # Lazy load InsightFace (~280 MB ONNX models). Ако fail-не → dlib fallback.
    try:
        insightface_app = await asyncio.to_thread(_get_insightface_app)
    except Exception as exc:
        _LOGGER.error("👤 Face loop CRASH at load: %s", exc, exc_info=True)
        insightface_app = None
    using_insightface = insightface_app is not None
    _LOGGER.info("👤 Face loop: InsightFace loaded=%s", using_insightface)

    if not using_insightface:
        try:
            import face_recognition  # noqa: F401 — проверка дали е инсталиран
        except ImportError:
            _LOGGER.warning("face_recognition не е инсталиран — face extraction изключен")
            return

    # One-time миграция на съществуващи crops към ArcFace 512-d (ако InsightFace ОК)
    if using_insightface:
        try:
            _LOGGER.info("👤 Face loop: starting migration check ...")
            await _migrate_faces_to_v2()
            _LOGGER.info("👤 Face loop: migration check done")
        except Exception as exc:
            _LOGGER.warning("Face v2 migration error: %s", exc, exc_info=True)

    _LOGGER.info(
        "👤 Face extraction: стартирал ✓ (engine=%s)",
        "InsightFace ArcFace 512-d" if using_insightface else "dlib HOG/CNN 128-d",
    )
    FACES_DIR.mkdir(parents=True, exist_ok=True)

    last_id_str = await asyncio.to_thread(_db_meta_get, "face_last_event_id", "0")
    last_id = int(last_id_str)

    while True:
        try:
            events = await asyncio.to_thread(_db_get_new_person_events, last_id)
            if events:
                for ev in events:
                    last_id = max(last_id, ev["id"])
                    try:
                        n = await loop.run_in_executor(None, _extract_faces_from_event, ev)
                        if n:
                            _LOGGER.info(
                                "👤 Face: %d лице(а) от event #%d (ch%d)",
                                n, ev["id"], ev["channel"],
                            )
                    except Exception as exc:
                        _LOGGER.warning("Face extract event #%d FAIL: %s", ev["id"], exc)
                await asyncio.to_thread(_db_meta_set, "face_last_event_id", str(last_id))
        except Exception as exc:
            _LOGGER.warning("Face extraction loop: %s", exc)

        await asyncio.sleep(30)


# ── lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(_db_init)
    await asyncio.to_thread(_db_seed_cameras_if_empty)
    await _cameras_refresh()
    tasks = [
        asyncio.create_task(_detection_loop()),
        asyncio.create_task(_shelly_detection_loop()),
        asyncio.create_task(_face_extraction_loop()),
        asyncio.create_task(_recording_loop()),
        asyncio.create_task(_recording_cleanup_loop()),
    ]
    yield
    for t in tasks:
        t.cancel()
    # Спираме FFmpeg recorder процесите чисто
    for ch, proc in list(_recording_procs.items()):
        try:
            proc.terminate()
        except Exception:
            pass


# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(title="Shelly NAS Hub", version="1.0.0", lifespan=lifespan)


# ── Authentication (single-user, session cookie) ───────────────────────────
SESSION_COOKIE_NAME = "homehub_session"
_auth_signer:        TimestampSigner | None = None


def _get_auth_signer() -> TimestampSigner | None:
    """Lazy-init signer от auth_settings.secret (или autogenerate)."""
    global _auth_signer
    if not auth_settings.enabled:
        return None
    if _auth_signer is not None:
        return _auth_signer
    secret = auth_settings.secret
    if not secret:
        # Auto-generate ако не е зададено — sessions invalidate при restart!
        import secrets as _secrets
        secret = _secrets.token_hex(32)
        _LOGGER.warning(
            "HOMEHUB_AUTH_SECRET not set — using ephemeral key. "
            "Sessions will invalidate on restart. Set in .env for persistence."
        )
    _auth_signer = TimestampSigner(secret)
    return _auth_signer


def _make_session_cookie(username: str) -> str:
    """Подписва username като session token. Връща string за Set-Cookie."""
    signer = _get_auth_signer()
    if signer is None:
        return ""
    return signer.sign(username.encode("utf-8")).decode("ascii")


def _verify_session_cookie(token: str | None) -> str | None:
    """Validate cookie token. Връща username ако е валиден, иначе None."""
    if not token:
        return None
    signer = _get_auth_signer()
    if signer is None:
        return None
    try:
        max_age = auth_settings.session_ttl_hours * 3600
        username = signer.unsign(token, max_age=max_age).decode("utf-8")
        return username
    except (BadSignature, SignatureExpired):
        return None
    except Exception:
        return None


# Auth scope: protect all camera UI + camera APIs. Login/logout/whoami и
# change-password се обработват от собствената си логика (без middleware).
_AUTH_PROTECTED_PAGES = {
    "/cameras.html",
    "/cameras-config.html",
    "/events.html",
    "/faces.html",
    "/recordings.html",
    "/shelly-cam.html",
    "/account.html",
}
# API prefixes WITHOUT trailing slash. Match-ват exact path или с "/..." суфикс.
_AUTH_PROTECTED_API_PREFIX = (
    "/api/camera",        # /api/camera, /api/camera/{ch}/snapshot|video|ws|events/bulk
    "/api/cameras",       # /api/cameras (CRUD), /api/cameras/storage,
                          # /api/cameras/{ch}/shelly-snapshot|shelly-rpc
    "/api/recordings",    # /api/recordings/cameras, /api/recordings/{ch}/timeline...
    "/api/faces",         # /api/faces/clusters, /api/faces/stats...
)


def _path_requires_auth(path: str) -> bool:
    """True ако даденият път е camera-related и трябва да require session."""
    if path in _AUTH_PROTECTED_PAGES:
        return True
    for prefix in _AUTH_PROTECTED_API_PREFIX:
        if path == prefix or path.startswith(prefix + "/"):
            return True
    # /api/auth/* — login/logout/whoami/change-password — handle вътрешно
    # /health — public (за health checks)
    return False


class AuthMiddleware(BaseHTTPMiddleware):
    """Проверява всеки request за валиден session cookie ако auth_enabled=True.

    - HTML pages → 302 redirect към /login.html?next=<original_path>
    - API requests → 401 JSON
    """
    async def dispatch(self, request: Request, call_next):
        if not auth_settings.enabled:
            return await call_next(request)

        path = request.url.path
        if not _path_requires_auth(path):
            return await call_next(request)

        token    = request.cookies.get(SESSION_COOKIE_NAME)
        username = _verify_session_cookie(token)
        if username is None:
            # Distinguish API request vs page request чрез Accept header.
            accept = request.headers.get("accept", "")
            wants_html = "text/html" in accept or path.endswith(".html") or path == "/"
            if wants_html:
                return RedirectResponse(
                    url=f"/login.html?next={path}",
                    status_code=302,
                )
            return JSONResponse(
                {"detail": "Authentication required"},
                status_code=401,
            )
        # Прокарваме username към endpoint-ите ако им трябва
        request.state.user = username
        return await call_next(request)


app.add_middleware(AuthMiddleware)


# ── Auth endpoints ─────────────────────────────────────────────────────────
class LoginBody(BaseModel):
    username: str
    password: str


@app.post("/api/auth/login")
async def auth_login(body: LoginBody, response: Response):
    """Validate username + password → set session cookie. Returns {ok, user}."""
    if not auth_settings.enabled:
        return {"ok": True, "user": None, "auth_disabled": True}

    if body.username != auth_settings.user:
        await asyncio.sleep(0.5)   # leak resistance
        raise HTTPException(401, "Invalid credentials")
    if not auth_settings.password_hash:
        raise HTTPException(503, "Auth misconfigured: no password hash set")
    try:
        ok = bcrypt.checkpw(
            body.password.encode("utf-8"),
            auth_settings.password_hash.encode("utf-8"),
        )
    except ValueError:
        ok = False
    if not ok:
        await asyncio.sleep(0.5)
        raise HTTPException(401, "Invalid credentials")

    token = _make_session_cookie(body.username)
    response.set_cookie(
        SESSION_COOKIE_NAME, token,
        max_age=auth_settings.session_ttl_hours * 3600,
        httponly=True, samesite="lax", secure=False,
    )
    _LOGGER.info("Auth: user '%s' logged in", body.username)
    return {"ok": True, "user": body.username}


@app.post("/api/auth/logout")
async def auth_logout(response: Response):
    response.delete_cookie(SESSION_COOKIE_NAME)
    return {"ok": True}


@app.get("/api/auth/whoami")
async def auth_whoami(request: Request):
    """Връща {user: ..., auth_enabled: bool}. Не изисква auth (по whitelist)."""
    if not auth_settings.enabled:
        return {"auth_enabled": False, "user": None}
    token    = request.cookies.get(SESSION_COOKIE_NAME)
    username = _verify_session_cookie(token)
    return {"auth_enabled": True, "user": username}


# ── Password change ─────────────────────────────────────────────────────────
# Runtime override: password_hash идва от env (`HOMEHUB_AUTH_PASSWORD_HASH`),
# но потребителят може да я смени през UI. Override-ът се пази в JSON файл
# (`/data/auth_override.json`), който се чете при startup и при всеки login
# (auth_settings.password_hash се обновява in-memory). Така `.env` остава clean.
AUTH_OVERRIDE_PATH = DATA_DIR / "auth_override.json"


def _load_auth_override() -> None:
    """Зарежда password_hash от auth_override.json (override на env)."""
    if not AUTH_OVERRIDE_PATH.exists():
        return
    try:
        data = json.loads(AUTH_OVERRIDE_PATH.read_text())
        ph = data.get("password_hash")
        if isinstance(ph, str) and ph:
            auth_settings.password_hash = ph
            _LOGGER.info("Auth: password_hash loaded from %s", AUTH_OVERRIDE_PATH)
    except Exception as exc:
        _LOGGER.error("Failed to load %s: %s", AUTH_OVERRIDE_PATH, exc)


def _save_auth_override(password_hash: str) -> None:
    """Persist password_hash в auth_override.json (chmod 600)."""
    AUTH_OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = AUTH_OVERRIDE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps({"password_hash": password_hash}))
    try:
        os.chmod(tmp, 0o600)
    except Exception:
        pass
    tmp.replace(AUTH_OVERRIDE_PATH)


_load_auth_override()


class ChangePasswordBody(BaseModel):
    current_password: str
    new_password:     str


@app.post("/api/auth/change-password")
async def auth_change_password(body: ChangePasswordBody, request: Request):
    """Смяна на собствената парола. Изисква active session + current_password."""
    if not auth_settings.enabled:
        raise HTTPException(400, "Auth is disabled — set HOMEHUB_AUTH_ENABLED=true first")

    token    = request.cookies.get(SESSION_COOKIE_NAME)
    username = _verify_session_cookie(token)
    if not username:
        raise HTTPException(401, "Not authenticated")

    if not auth_settings.password_hash:
        raise HTTPException(503, "Auth misconfigured: no password hash set")

    try:
        ok = bcrypt.checkpw(
            body.current_password.encode("utf-8"),
            auth_settings.password_hash.encode("utf-8"),
        )
    except ValueError:
        ok = False
    if not ok:
        await asyncio.sleep(0.5)
        raise HTTPException(401, "Current password is incorrect")

    if len(body.new_password) < 6:
        raise HTTPException(400, "New password must be at least 6 characters")
    if body.new_password == body.current_password:
        raise HTTPException(400, "New password must differ from current")

    new_hash = bcrypt.hashpw(body.new_password.encode("utf-8"), bcrypt.gensalt()).decode()
    auth_settings.password_hash = new_hash
    try:
        _save_auth_override(new_hash)
    except Exception as exc:
        _LOGGER.error("Failed to persist new password hash: %s", exc)
        raise HTTPException(500, "Failed to persist password change")

    _LOGGER.info("Auth: user '%s' changed password", username)
    return {"ok": True}


# ── Health endpoint ────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "ok":           True,
        "recording":    recording_settings.enabled,
        "detection":    detect_settings.enabled,
        "auth_enabled": auth_settings.enabled,
    }


# ── NVR / Camera ────────────────────────────────────────────────────────────
_nvr_snap_cache:  dict[int, tuple[float, bytes]]  = {}   # ch → (ts, jpeg)
_detect_armed:    dict[int, bool]                 = {}   # ch → armed (зарежда се при старт)
_detect_conf:     dict[int, float | None]         = {}   # ch → min_confidence (None = глобална)

# ── Object tracking (per-camera, IoU-based) ────────────────────────────────
# Как работи:
#   - Всеки засечен обект получава стабилен `track_id` (per camera).
#   - Между кадри обектите се мач-ват по IoU ≥ TRACK_IOU_THRESHOLD И същия label.
#   - Track се smята за "същия обект" докато се вижда поне веднъж в TRACK_TIMEOUT_S.
#   - След пауза > TRACK_TIMEOUT_S → нов track_id → НОВ event (върнал се обект).
#   - Track confirmation: track става "потвърден" чак когато е видян в N последователни
#     кадъра (default 2). До тогава е "tentative" — не генерира event и се изтрива
#     ако не се появи отново в TENTATIVE_TIMEOUT_S. Това отсява YOLO false positives
#     които обикновено се появяват само в 1 кадър.
# Резултат: паркирана кола = един track = ЕДИН event докато не изчезне за 5 мин.
TRACK_IOU_THRESHOLD:  float = 0.30
TRACK_TIMEOUT_S:      float = 300.0   # 5 минути за потвърдени tracks
TENTATIVE_TIMEOUT_S:  float = 15.0    # 15 секунди за непотвърдени (filter за false positives)

# ch → {track_id: {"label", "box":[x1,y1,x2,y2], "first_seen", "last_seen", "seen_count", "confirmed"}}
_track_state:    dict[int, dict[int, dict]] = {}
_track_next_id:  dict[int, int]             = {}   # per-camera autoincrement


def _iou(a: list[int], b: list[int]) -> float:
    """IoU за два box-а [x1,y1,x2,y2]."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih   = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter    = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
    area_b = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _track_update(ch: int, det_boxes: list[dict], now_ts: float) -> list[dict]:
    """Обновява track state за камера и връща за всеки detection: track_id, is_new.

    det_boxes: list of {"label": str, "box": [x1,y1,x2,y2]}.
    Връща: list of {"label", "box", "track_id", "is_new"}.
      is_new = True САМО когато track достигне статус "confirmed" (видян в
      `confirm_frames` последователни кадъра). Tentative tracks (потенциални
      false positives) не дават event.
    """
    state = _track_state.setdefault(ch, {})

    # Изтрий tracks с различен timeout за potvrden vs tentative
    expired = []
    for tid, t in state.items():
        timeout = TRACK_TIMEOUT_S if t.get("confirmed") else TENTATIVE_TIMEOUT_S
        if (now_ts - t["last_seen"]) > timeout:
            expired.append(tid)
    for tid in expired:
        del state[tid]

    confirm_n: int = max(1, int(detect_settings.confirm_frames))

    used_track_ids: set[int] = set()
    result: list[dict] = []

    for det in det_boxes:
        label = det["label"]
        box   = det["box"]

        best_tid: int | None = None
        best_iou: float      = TRACK_IOU_THRESHOLD
        for tid, t in state.items():
            if tid in used_track_ids:
                continue
            if t["label"] != label:
                continue
            iou = _iou(box, t["box"])
            if iou > best_iou:
                best_iou = iou
                best_tid = tid

        is_new = False
        if best_tid is None:
            best_tid = _track_next_id.get(ch, 1)
            _track_next_id[ch] = best_tid + 1
            state[best_tid] = {
                "label":      label,
                "box":        box,
                "first_seen": now_ts,
                "last_seen":  now_ts,
                "seen_count": 1,
                "confirmed":  (confirm_n <= 1),
            }
            # Ако confirm_n==1 → потвърден от 1ва детекция → веднага event
            is_new = state[best_tid]["confirmed"]
        else:
            t = state[best_tid]
            was_tentative   = not t["confirmed"]
            t["box"]        = box
            t["last_seen"]  = now_ts
            t["seen_count"] = t.get("seen_count", 1) + 1
            if was_tentative and t["seen_count"] >= confirm_n:
                t["confirmed"] = True
                is_new         = True   # Промоция → генерирай event

        used_track_ids.add(best_tid)
        result.append({"label": label, "box": box, "track_id": best_tid, "is_new": is_new})

    return result


def _nvr_valid_channels() -> list[int]:
    """NVR Dahua канали (type='nvr_dahua' и enabled), от DB cache."""
    return sorted([
        ch for ch, c in _cameras_get_all().items()
        if c["type"] == "nvr_dahua" and c["enabled"]
    ])


def _all_valid_channels() -> list[int]:
    """NVR Dahua + RTSP/Reolink канали (type in nvr_dahua/rtsp_generic), от DB."""
    return sorted([
        ch for ch, c in _cameras_get_all().items()
        if c["type"] in ("nvr_dahua", "rtsp_generic") and c["enabled"]
    ])


def _ch_display_name(ch: int) -> str | None:
    """Връща UI името за канал от camera registry."""
    cam = _get_camera(ch)
    return cam["name"] if cam else None


@app.get("/api/camera/list")
async def camera_list():
    """Списък на ВСИЧКИ enabled камери (NVR + RTSP + Shelly), сортирани по channel."""
    result = [
        {"channel": cam["channel"], "name": cam["name"], "type": cam["type"]}
        for cam in _cameras_get_all().values()
        if cam["enabled"]
    ]
    return {"channels": sorted(result, key=lambda r: r["channel"])}


# ── Camera CRUD ─────────────────────────────────────────────────────────────
class CameraCreateBody(BaseModel):
    """Schema за добавяне/редакция на камера.

    type: 'nvr_dahua' | 'rtsp_generic' | 'shelly'
    config: type-specific dict — виж _db_seed_cameras_if_empty за полетата.
    retention_days: per-camera override; None → use global RECORDING_RETENTION_DAYS.
    """
    channel:        int
    type:           str
    name:           str
    enabled:        bool = True
    has_audio:      bool = False
    config:         dict
    retention_days: int | None = None


class CameraUpdateBody(BaseModel):
    """Всички полета са optional → PATCH-style update."""
    channel:        int | None = None
    type:           str | None = None
    name:           str | None = None
    enabled:        bool | None = None
    has_audio:      bool | None = None
    config:         dict | None = None
    retention_days: int | None = None


def _validate_camera_config(ctype: str, config: dict) -> None:
    """Хвърля HTTPException(400) ако type-specific конфигурацията е невалидна."""
    if ctype not in ("nvr_dahua", "rtsp_generic", "shelly"):
        raise HTTPException(400, f"Unknown camera type: {ctype}")

    if ctype == "nvr_dahua":
        for f in ("host", "user", "password"):
            if not config.get(f):
                raise HTTPException(400, f"NVR Dahua: missing required '{f}'")
        if "channel_id" not in config:
            raise HTTPException(400, "NVR Dahua: missing 'channel_id'")

    elif ctype == "rtsp_generic":
        if not config.get("rtsp_main"):
            raise HTTPException(400, "RTSP camera: missing 'rtsp_main' URL")

    elif ctype == "shelly":
        if not config.get("ip"):
            raise HTTPException(400, "Shelly camera: missing 'ip'")


@app.get("/api/cameras")
async def cameras_list_all():
    """Връща ВСИЧКИ камери (включително disabled) с пълен config за CRUD UI."""
    rows = await asyncio.to_thread(_db_list_cameras)
    return {"cameras": rows}


@app.get("/api/cameras/storage")
async def cameras_storage():
    """Per-camera storage info + retention estimate.

    Връща {global_retention_days, disk: {...}, cameras: [
       {channel, name, retention_days (override), effective_retention_days,
        size_bytes, days_present, avg_bytes_per_day, estimated_size_bytes}
    ]}

    Идея:
    - avg_bytes_per_day = size_bytes / max(1, дни_със_записи)
    - estimated_size_bytes = avg × effective_retention (колко място ще заема
      когато retention периодът се напълни напълно)
    """
    # Refresh disk usage cache ако е изтекъл (същата логика като recordings/cameras)
    if (time.monotonic() - _disk_usage_cache_ts) > DISK_USAGE_TTL_S:
        await asyncio.to_thread(_refresh_disk_cache_sync)

    def _build():
        global_keep = recording_settings.retention_days
        out = []
        for cam in _cameras_get_all().values():
            if not cam["enabled"]:
                continue
            ch = cam["channel"]
            disk_info = _disk_usage_cache.get(ch, {"days_present": [], "size_bytes": 0})
            size = disk_info["size_bytes"]
            days = len(disk_info["days_present"])
            override = cam.get("retention_days")
            effective = int(override) if override else global_keep
            avg_per_day = size / days if days > 0 else 0.0
            estimated = int(avg_per_day * effective)
            out.append({
                "channel":                  ch,
                "name":                     cam["name"],
                "type":                     cam["type"],
                "retention_days":           override,           # None → global
                "effective_retention_days": effective,
                "global_retention_days":    global_keep,
                "size_bytes":               size,
                "days_present":             days,
                "avg_bytes_per_day":        int(avg_per_day),
                "estimated_size_bytes":     estimated,
            })
        out.sort(key=lambda r: r["channel"])

        try:
            ds = shutil.disk_usage(recording_settings.path)
            disk = {
                "total_bytes": ds.total,
                "used_bytes":  ds.used,
                "free_bytes":  ds.free,
                "used_pct":    round(ds.used / ds.total * 100, 1),
                "limit_pct":   recording_settings.disk_limit_pct,
            }
        except Exception:
            disk = None

        total_estimated = sum(c["estimated_size_bytes"] for c in out)
        return {
            "global_retention_days": global_keep,
            "disk":                  disk,
            "cameras":               out,
            "total_estimated_bytes": total_estimated,
        }

    return await asyncio.to_thread(_build)


@app.post("/api/cameras")
async def cameras_create(body: CameraCreateBody):
    """Добавя нова камера. Channel трябва да е unique."""
    _validate_camera_config(body.type, body.config)

    if await asyncio.to_thread(_db_get_camera_by_channel, body.channel):
        raise HTTPException(409, f"Camera with channel {body.channel} already exists")

    try:
        cam_id = await asyncio.to_thread(
            _db_insert_camera, body.channel, body.type, body.name,
            body.config, body.has_audio, body.enabled, body.retention_days,
        )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(409, f"Channel conflict: {exc}")

    await _cameras_refresh()
    _LOGGER.info("Camera added: ch=%d type=%s name=%s id=%d",
                 body.channel, body.type, body.name, cam_id)

    new_cam = await asyncio.to_thread(_db_get_camera_by_id, cam_id)
    return {"camera": new_cam}


@app.put("/api/cameras/{cam_id}")
async def cameras_update(cam_id: int, body: CameraUpdateBody):
    """Редакция на камера. Полетата които не са дадени остават непроменени."""
    existing = await asyncio.to_thread(_db_get_camera_by_id, cam_id)
    if not existing:
        raise HTTPException(404, "Camera not found")

    fields = body.model_dump(exclude_unset=True)

    # Validate ако се променя type или config
    if fields:
        new_type   = fields.get("type", existing["type"])
        new_config = fields.get("config", existing["config"])
        if "type" in fields or "config" in fields:
            _validate_camera_config(new_type, new_config)

        # Ако се променя channel — трябва да е unique
        if "channel" in fields and fields["channel"] != existing["channel"]:
            other = await asyncio.to_thread(_db_get_camera_by_channel, fields["channel"])
            if other and other["id"] != cam_id:
                raise HTTPException(409, f"Channel {fields['channel']} вече е заето")

    ok = await asyncio.to_thread(_db_update_camera, cam_id, fields)
    if not ok:
        raise HTTPException(400, "No changes applied")

    await _cameras_refresh()
    _LOGGER.info("Camera updated: id=%d fields=%s", cam_id, list(fields.keys()))

    updated = await asyncio.to_thread(_db_get_camera_by_id, cam_id)
    return {"camera": updated}


def _delete_camera_recordings(channel: int) -> dict:
    """Триене на recordings + events + face crops за камера. Връща статистика."""
    stats = {"recordings_dirs": 0, "events": 0, "event_files": 0,
             "face_crops": 0}

    # 1. Recordings — целия /recordings/{ch}/
    cam_dir = _ch_recording_dir(channel)
    if cam_dir.exists():
        try:
            shutil.rmtree(cam_dir, ignore_errors=True)
            stats["recordings_dirs"] = 1
        except Exception as exc:
            _LOGGER.warning("delete_camera_recordings ch=%d: %s", channel, exc)

    # 2. Detection events + JPEG-ите
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, image_path FROM detection_events WHERE channel = ?",
            (channel,),
        ).fetchall()
        for (_eid, path) in rows:
            if path:
                try:
                    Path(path).unlink(missing_ok=True)
                    stats["event_files"] += 1
                except Exception:
                    pass
        cur = conn.execute(
            "DELETE FROM detection_events WHERE channel = ?", (channel,),
        )
        stats["events"] = cur.rowcount

        # 3. Face crops от тази камера
        crop_rows = conn.execute(
            "SELECT image_path FROM face_crops WHERE channel = ?",
            (channel,),
        ).fetchall()
        for (path,) in crop_rows:
            if path:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    pass
        cur = conn.execute(
            "DELETE FROM face_crops WHERE channel = ?", (channel,),
        )
        stats["face_crops"] = cur.rowcount

        # Изчистване на detection_config row
        conn.execute("DELETE FROM detection_config WHERE channel = ?", (channel,))

    return stats


@app.delete("/api/cameras/{cam_id}")
async def cameras_delete(cam_id: int, delete_recordings: bool = False):
    """Триене на камера. По подразбиране запазва recordings/events/faces.

    Query: ?delete_recordings=true → трие също записите, events, face crops.
    """
    cam = await asyncio.to_thread(_db_get_camera_by_id, cam_id)
    if not cam:
        raise HTTPException(404, "Camera not found")

    channel = cam["channel"]

    # Спираме recorder ако работи
    if channel in _recording_procs:
        try:
            await _stop_recorder(channel)
        except Exception as exc:
            _LOGGER.warning("stop recorder при delete cam%d: %s", channel, exc)

    stats = {"deleted": True, "recordings_kept": True}
    if delete_recordings:
        stats = await asyncio.to_thread(_delete_camera_recordings, channel)
        stats["deleted"] = True
        stats["recordings_kept"] = False

    await asyncio.to_thread(_db_delete_camera, cam_id)
    await _cameras_refresh()
    _LOGGER.info("Camera deleted: id=%d ch=%d delete_recordings=%s",
                 cam_id, channel, delete_recordings)

    return {"camera_id": cam_id, "channel": channel, **stats}


@app.get("/api/cameras/{channel}/shelly-snapshot")
async def cameras_shelly_snapshot(channel: int):
    """Backend proxy за Shelly /camera/0/snapshot. Работи за всякакъв channel
    без да изисква hardcoded nginx config. Заменя /shelly-snapshot{,-2,-3}."""
    cam = _get_camera(channel)
    if not cam or cam["type"] != "shelly":
        raise HTTPException(404, "Shelly camera not found")
    ip = cam["config"].get("ip", "")
    if not ip:
        raise HTTPException(503, "Shelly IP not configured")

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(8.0)) as client:
            r = await client.get(f"http://{ip}/camera/0/snapshot")
            if r.status_code != 200:
                raise HTTPException(502, f"Shelly returned HTTP {r.status_code}")
            return Response(
                content=r.content,
                media_type="image/jpeg",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Shelly fetch error: {exc}")


@app.api_route(
    "/api/cameras/{channel}/shelly-rpc/{rpc_path:path}",
    methods=["GET", "POST", "OPTIONS"],
)
async def cameras_shelly_rpc(channel: int, rpc_path: str, request: Request):
    """Backend proxy за Shelly /rpc/* — заменя /shelly-rpc{,-2,-3}/ от nginx.
    Поддържа GET/POST с body (JSON-RPC) — за WebRTC signaling."""
    cam = _get_camera(channel)
    if not cam or cam["type"] != "shelly":
        raise HTTPException(404, "Shelly camera not found")
    ip = cam["config"].get("ip", "")
    if not ip:
        raise HTTPException(503, "Shelly IP not configured")

    if request.method == "OPTIONS":
        return Response(status_code=204, headers={
            "Access-Control-Allow-Origin":  "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        })

    target_url = f"http://{ip}/rpc/{rpc_path}"
    body       = await request.body()
    headers    = {"Content-Type": request.headers.get("content-type", "application/json")}
    params     = dict(request.query_params)

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            if request.method == "GET":
                r = await client.get(target_url, params=params, headers=headers)
            else:
                r = await client.post(target_url, params=params, content=body, headers=headers)
            return Response(
                content=r.content,
                status_code=r.status_code,
                media_type=r.headers.get("content-type", "application/json"),
            )
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Shelly RPC error: {exc}")


@app.post("/api/cameras/{cam_id}/test")
async def cameras_test(cam_id: int):
    """Тества връзката със камера: snapshot fetch.
    Връща {ok: bool, message: str, latency_ms?: int}."""
    cam = await asyncio.to_thread(_db_get_camera_by_id, cam_id)
    if not cam:
        raise HTTPException(404, "Camera not found")

    cfg = cam["config"]
    t0 = time.monotonic()

    if cam["type"] == "shelly":
        ip = cfg.get("ip", "")
        if not ip:
            return {"ok": False, "message": "No IP configured"}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                r = await client.get(f"http://{ip}/camera/0/snapshot")
                ms = int((time.monotonic() - t0) * 1000)
                if r.status_code == 200 and len(r.content) > 5_000:
                    return {"ok": True, "message": f"Shelly snapshot OK ({len(r.content)} B)",
                            "latency_ms": ms}
                return {"ok": False, "message": f"HTTP {r.status_code}, body={len(r.content)} B",
                        "latency_ms": ms}
        except Exception as exc:
            return {"ok": False, "message": f"Connection error: {exc}"}

    if cam["type"] == "rtsp_generic":
        snap_url = cfg.get("snap_url", "")
        if snap_url:
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(8.0), verify=False) as client:
                    r = await client.get(snap_url)
                    ms = int((time.monotonic() - t0) * 1000)
                    if r.status_code == 200 and len(r.content) > 5_000:
                        return {"ok": True, "message": f"HTTP snapshot OK ({len(r.content)} B)",
                                "latency_ms": ms}
                    return {"ok": False, "message": f"HTTP {r.status_code}", "latency_ms": ms}
            except Exception as exc:
                return {"ok": False, "message": f"HTTP error: {exc}"}
        # Fallback: ffprobe RTSP
        url = cfg.get("rtsp_main", "")
        if not url:
            return {"ok": False, "message": "No URL configured"}
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffprobe", "-v", "error", "-rtsp_transport", "tcp",
                "-show_streams", "-of", "json", url,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            ms = int((time.monotonic() - t0) * 1000)
            if proc.returncode == 0 and b'"streams"' in stdout:
                return {"ok": True, "message": "RTSP probe OK", "latency_ms": ms}
            err = stderr.decode("utf-8", errors="replace")[:200]
            return {"ok": False, "message": f"ffprobe failed: {err}", "latency_ms": ms}
        except Exception as exc:
            return {"ok": False, "message": f"ffprobe error: {exc}"}

    if cam["type"] == "nvr_dahua":
        host = cfg.get("host") or nvr_settings.host
        user = cfg.get("user") or nvr_settings.user
        pwd  = cfg.get("password") or nvr_settings.password
        ch_id = cfg.get("channel_id", cam["channel"])
        if not (host and user and pwd):
            return {"ok": False, "message": "Missing host/user/password"}
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(8.0),
                auth=httpx.DigestAuth(user, pwd),
            ) as client:
                r = await client.get(f"http://{host}/cgi-bin/snapshot.cgi?channel={ch_id}")
                ms = int((time.monotonic() - t0) * 1000)
                if r.status_code == 200 and len(r.content) > 5_000:
                    return {"ok": True, "message": f"NVR snapshot OK ({len(r.content)} B)",
                            "latency_ms": ms}
                return {"ok": False, "message": f"HTTP {r.status_code}", "latency_ms": ms}
        except Exception as exc:
            return {"ok": False, "message": f"Connection error: {exc}"}

    return {"ok": False, "message": f"Unknown type: {cam['type']}"}


@app.get("/api/camera/events")
async def camera_events(
    channel: int | None = Query(default=None),
    limit:   int        = Query(default=100, le=1000),
):
    rows = await asyncio.to_thread(_db_get_detections, channel, limit)
    return {"events": rows}


@app.get("/api/camera/events/{event_id}/image")
async def camera_event_image(event_id: int):
    def _get():
        with sqlite3.connect(DB_PATH) as conn:
            return conn.execute(
                "SELECT image_path FROM detection_events WHERE id=?", (event_id,)
            ).fetchone()

    row = await asyncio.to_thread(_get)
    if not row or not row[0]:
        raise HTTPException(404, "Не е намерен")
    p = Path(row[0])
    if not p.exists():
        raise HTTPException(404, "Файлът не съществува")
    return FileResponse(p, media_type="image/jpeg")


# ── 24/7 Recording API endpoints ────────────────────────────────────────────

# ── Disk usage cache (за recordings/cameras endpoint) ──────────────────────
# Сканирането на recordings/{ch}/YYYYMMDD/HH/MM.mp4 за всички камери може да
# отнеме 30-120 секунди при 7+ дни данни (хиляди файлове × stat() syscalls).
# Cache-ваме на всеки 5 мин — endpoint-ът ползва cached values за бърз отговор.
_disk_usage_cache:    dict[int, dict] = {}
_disk_usage_cache_ts: float           = 0.0
DISK_USAGE_TTL_S:     float           = 300.0   # 5 минути


def _scan_camera_disk_usage(ch: int) -> dict:
    """Scans /recordings/{ch}/ directory tree. Връща {days_present, size_bytes}."""
    cam_dir = _ch_recording_dir(ch)
    days_present: list[str] = []
    total_size = 0
    if cam_dir.exists():
        for day_dir in sorted(cam_dir.iterdir()):
            if not (day_dir.is_dir() and len(day_dir.name) == 8 and day_dir.name.isdigit()):
                continue
            days_present.append(day_dir.name)
            for f in day_dir.rglob("*.mp4"):
                try:
                    total_size += f.stat().st_size
                except Exception:
                    pass
    return {"days_present": days_present, "size_bytes": total_size}


def _refresh_disk_cache_sync() -> None:
    """Re-scans всички enabled cameras и попълва _disk_usage_cache."""
    global _disk_usage_cache_ts
    new_cache: dict[int, dict] = {}
    for cam in _cameras_get_all().values():
        if not cam["enabled"]:
            continue
        try:
            new_cache[cam["channel"]] = _scan_camera_disk_usage(cam["channel"])
        except Exception as exc:
            _LOGGER.warning("disk usage scan ch=%d: %s", cam["channel"], exc)
    _disk_usage_cache.clear()
    _disk_usage_cache.update(new_cache)
    _disk_usage_cache_ts = time.monotonic()


@app.get("/api/recordings/cameras")
async def recordings_cameras_list():
    """Списък камери с recording status + disk info (cached)."""
    # Lazy refresh ако TTL изтекъл (или празен cache)
    if (time.monotonic() - _disk_usage_cache_ts) > DISK_USAGE_TTL_S:
        await asyncio.to_thread(_refresh_disk_cache_sync)

    def _build():
        out = []
        for cam in _cameras_get_all().values():
            if not cam["enabled"]:
                continue
            ch = cam["channel"]
            disk_info = _disk_usage_cache.get(ch, {"days_present": [], "size_bytes": 0})
            stats = _recording_stats.get(ch, {})
            proc  = _recording_procs.get(ch)
            kind = {"nvr_dahua": "nvr", "rtsp_generic": "extra", "shelly": "shelly"}.get(
                cam["type"], cam["type"]
            )
            out.append({
                "channel":         ch,
                "name":            cam["name"],
                "kind":            kind,
                "supported":       True,
                "armed":           _recording_armed.get(ch, True),
                "active":          proc is not None and proc.returncode is None,
                "started_at":      stats.get("started_at"),
                "last_segment_ts": stats.get("last_segment_ts"),
                "restarts":        stats.get("restarts", 0),
                "days_recorded":   disk_info["days_present"],
                "size_bytes":      disk_info["size_bytes"],
            })
        out.sort(key=lambda r: r["channel"])

        try:
            stat = shutil.disk_usage(recording_settings.path)
            disk = {
                "total_gb":    round(stat.total / 1e9, 1),
                "used_gb":     round(stat.used  / 1e9, 1),
                "free_gb":     round(stat.free  / 1e9, 1),
                "used_pct":    round(stat.used / stat.total * 100, 1),
                "limit_pct":   recording_settings.disk_limit_pct,
            }
        except Exception:
            disk = None

        return {
            "enabled":        recording_settings.enabled,
            "retention_days": recording_settings.retention_days,
            "segment_s":      recording_settings.segment_s,
            "quality":        recording_settings.quality,
            "audio":          recording_settings.audio,
            "cameras":        out,
            "disk":           disk,
        }

    return await asyncio.to_thread(_build)


@app.get("/api/recordings/{ch}/timeline")
async def recordings_timeline(ch: int, date: str = Query(..., regex=r"^\d{4}-\d{2}-\d{2}$")):
    """За дадена камера + ден връща списък налични segments + всички detection events."""
    def _build():
        # date YYYY-MM-DD → дир YYYYMMDD
        try:
            day_obj = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(400, "Грешен формат на date")

        day_str = day_obj.strftime("%Y%m%d")
        cam_dir = _ch_recording_dir(ch) / day_str

        segments: list[dict] = []
        if cam_dir.exists():
            for hour_dir in sorted(cam_dir.iterdir()):
                if not hour_dir.is_dir() or len(hour_dir.name) != 2:
                    continue
                hh = hour_dir.name
                for mp4 in sorted(hour_dir.glob("*.mp4")):
                    mm = mp4.stem  # MM
                    if not mm.isdigit() or len(mm) != 2:
                        continue
                    try:
                        size = mp4.stat().st_size
                    except Exception:
                        size = 0
                    if size < 100:  # пропусни празни/corrupted segments
                        continue
                    segments.append({
                        "ts":    f"{date}T{hh}:{mm}:00",
                        "hour":  int(hh),
                        "min":   int(mm),
                        "size":  size,
                        "url":   f"/api/recordings/{ch}/segment?ts={date}T{hh}:{mm}:00",
                    })

        # Detection events за същия ден на тази камера
        events: list[dict] = []
        try:
            day_start = day_obj.strftime("%Y-%m-%dT00:00:00")
            day_end   = (day_obj + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00")
            with sqlite3.connect(DB_PATH) as conn:
                rows = conn.execute(
                    "SELECT id, detected_at, labels, faces "
                    "FROM detection_events "
                    "WHERE channel=? AND detected_at>=? AND detected_at<? "
                    "ORDER BY detected_at ASC",
                    (ch, day_start, day_end),
                ).fetchall()
                for r in rows:
                    try:
                        labels = json.loads(r[2]) if r[2] else []
                    except Exception:
                        labels = []
                    try:
                        faces = json.loads(r[3]) if r[3] else []
                    except Exception:
                        faces = []
                    events.append({
                        "id":     r[0],
                        "ts":     r[1],
                        "labels": labels,
                        "faces":  faces,
                    })
        except Exception as exc:
            _LOGGER.debug("timeline events: %s", exc)

        return {
            "channel":  ch,
            "date":     date,
            "segments": segments,
            "events":   events,
        }

    return await asyncio.to_thread(_build)


@app.get("/api/recordings/{ch}/segment")
async def recordings_segment(
    ch: int,
    ts: str = Query(..., regex=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?$"),
):
    """Стрийма конкретен segment (1-мин MP4) за дадена камера и timestamp.

    `ts` в ISO формат `YYYY-MM-DDTHH:MM[:SS]` — секундите се игнорират
    (segment-ите са 1-минутни alligned до minute boundary).
    """
    try:
        ts_short = ts[:16]  # YYYY-MM-DDTHH:MM
        dt = datetime.strptime(ts_short, "%Y-%m-%dT%H:%M")
    except ValueError:
        raise HTTPException(400, "Грешен формат на ts")

    seg_path = (
        _ch_recording_dir(ch) / dt.strftime("%Y%m%d") / dt.strftime("%H") / f"{dt.minute:02d}.mp4"
    )
    if not seg_path.exists():
        raise HTTPException(404, "Сегментът не е намерен")

    return FileResponse(seg_path, media_type="video/mp4", filename=f"ch{ch}_{ts_short}.mp4")


class RecordingArmBody(BaseModel):
    armed: bool


@app.post("/api/recordings/{ch}/arm")
async def recordings_arm(ch: int, body: RecordingArmBody):
    """Enable/disable continuous recording per камера."""
    if not _recording_ch_names.get(ch):
        raise HTTPException(404, f"Камера {ch} не е конфигурирана за recording")

    _recording_armed[ch] = bool(body.armed)
    await asyncio.to_thread(
        _db_meta_set, f"recording_armed_{ch}", "1" if body.armed else "0",
    )

    if body.armed:
        ok = await _start_recorder(ch)
        return {"channel": ch, "armed": True, "started": ok}
    else:
        await _stop_recorder(ch)
        return {"channel": ch, "armed": False, "stopped": True}


@app.get("/api/recordings/{ch}/live.m3u8")
async def recordings_live_playlist(ch: int):
    """HLS playlist за live streaming (споделен със recorder-а — без 2-ри RTSP connection).

    Връща 404 ако recording не е armed за тази камера → клиентът трябва да fallback
    към MJPEG (`/api/camera/{ch}/video`).
    """
    if not _recording_armed.get(ch, False):
        raise HTTPException(404, "Recording не е активен за тази камера")
    pl = _ch_recording_dir(ch) / "live" / "live.m3u8"
    if not pl.exists():
        raise HTTPException(404, "HLS playlist още не е готов (изчакай 5 сек)")
    # No-cache headers за rolling playlist
    return FileResponse(
        pl,
        media_type="application/vnd.apple.mpegurl",
        headers={"Cache-Control": "no-cache, no-store"},
    )


@app.get("/api/recordings/{ch}/live/{filename}")
async def recordings_live_segment(ch: int, filename: str):
    """HLS .ts segment файл (relative към playlist-а)."""
    # Sanity check на filename — само live*.ts
    if not (filename.startswith("live") and filename.endswith(".ts")):
        raise HTTPException(400, "Грешен HLS segment filename")
    seg = _ch_recording_dir(ch) / "live" / filename
    if not seg.exists():
        raise HTTPException(404, "Segment не е намерен")
    return FileResponse(
        seg, media_type="video/mp2t",
        headers={"Cache-Control": "max-age=10"},
    )


@app.get("/api/recordings/{ch}/{filename}")
async def recordings_live_segment_compat(ch: int, filename: str):
    """HLS .ts segments на сибling-path до playlist-а.

    Playlist (`/api/recordings/{ch}/live.m3u8`) сервира segments като relative
    URLs (`live3969.ts`), които hls.js resolve-ва спрямо playlist directory →
    `/api/recordings/{ch}/live3969.ts` (БЕЗ `/live/` префикс). Този endpoint
    proxy-ра към `recordings_live_segment` за всички `liveN.ts` заявки.
    """
    if not (filename.startswith("live") and filename.endswith(".ts")):
        raise HTTPException(404)
    return await recordings_live_segment(ch, filename)


class DetectArmBody(BaseModel):
    armed: bool


class DeleteEventsBody(BaseModel):
    ids: list[int]


@app.delete("/api/camera/events")
async def camera_events_delete(body: DeleteEventsBody):
    """Изтрива избрани detection events (по id) + JPEG файловете им."""
    if not body.ids:
        return {"deleted": 0}

    def _delete(ids: list[int]) -> int:
        with sqlite3.connect(DB_PATH) as conn:
            placeholders = ",".join("?" * len(ids))
            rows = conn.execute(
                f"SELECT id, image_path FROM detection_events WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
            for _eid, p in rows:
                try:
                    if p:
                        Path(p).unlink(missing_ok=True)
                except Exception:
                    pass
            conn.execute(
                f"DELETE FROM detection_events WHERE id IN ({placeholders})", ids
            )
            return len(rows)

    deleted = await asyncio.to_thread(_delete, body.ids)
    return {"deleted": deleted}


@app.delete("/api/camera/events/bulk")
async def camera_events_bulk_delete(
    older_than_days: int | None = None,
    channel:         int | None = None,
    delete_all:      bool       = False,
):
    """Bulk delete на detection events.

    Query params:
      ?older_than_days=N — само events по-стари от N дни
      ?channel=CH         — само от канал CH (combined с older_than_days)
      ?delete_all=true    — ВСИЧКИ events (изисква експлицитно)

    Поне един от older_than_days/delete_all трябва да е зададен.
    """
    if not delete_all and (older_than_days is None or older_than_days < 0):
        raise HTTPException(400, "Provide ?older_than_days=N or ?delete_all=true")

    def _bulk():
        with sqlite3.connect(DB_PATH) as conn:
            where = []
            params: list = []
            if older_than_days is not None and older_than_days >= 0:
                cutoff = (datetime.now(timezone.utc)
                          - timedelta(days=older_than_days)).isoformat()
                where.append("detected_at < ?")
                params.append(cutoff)
            if channel is not None:
                where.append("channel = ?")
                params.append(channel)
            sql_where = (" WHERE " + " AND ".join(where)) if where else ""
            rows = conn.execute(
                f"SELECT id, image_path FROM detection_events{sql_where}",
                params,
            ).fetchall()
            files_deleted = 0
            for _eid, p in rows:
                if p:
                    try:
                        Path(p).unlink(missing_ok=True)
                        files_deleted += 1
                    except Exception:
                        pass
            conn.execute(f"DELETE FROM detection_events{sql_where}", params)
            return len(rows), files_deleted

    deleted, files = await asyncio.to_thread(_bulk)
    _LOGGER.info("Bulk events delete: events=%d files=%d days=%s ch=%s all=%s",
                 deleted, files, older_than_days, channel, delete_all)
    return {"deleted_events": deleted, "deleted_files": files}


@app.delete("/api/faces/clusters/bulk")
async def faces_clusters_bulk_delete(
    keep_named: bool       = True,
    channel:    int | None = None,
):
    """Bulk delete на face clusters + техните crops.

    Query:
      ?keep_named=true (default) — пази клъстери с име, трие само unknown
      ?keep_named=false           — трие ВСИЧКИ (named + unnamed)
      ?channel=CH                 — само crops от канал CH (но цели clusters
                                    се трият само ако всички им crops са от него)
    """
    def _bulk():
        with sqlite3.connect(DB_PATH) as conn:
            # 1. Намираме кандидатите за triene
            sql = "SELECT id, name FROM face_clusters"
            params: list = []
            if keep_named:
                sql += " WHERE name IS NULL"
            cluster_rows = conn.execute(sql, params).fetchall()

            deleted_clusters = 0
            deleted_crops    = 0
            deleted_files    = 0

            for cid, _cname in cluster_rows:
                # Ако филтрираме по канал — пропускаме clusters с crops от
                # други канали (за да не загубим миксирани идентичности).
                if channel is not None:
                    other_ch = conn.execute(
                        "SELECT COUNT(*) FROM face_crops "
                        "WHERE cluster_id = ? AND channel != ?",
                        (cid, channel),
                    ).fetchone()[0]
                    if other_ch > 0:
                        continue

                crops = conn.execute(
                    "SELECT image_path FROM face_crops WHERE cluster_id = ?",
                    (cid,),
                ).fetchall()
                for (path,) in crops:
                    if path:
                        try:
                            Path(path).unlink(missing_ok=True)
                            deleted_files += 1
                        except Exception:
                            pass
                conn.execute("DELETE FROM face_crops WHERE cluster_id = ?", (cid,))
                deleted_crops += len(crops)
                conn.execute("DELETE FROM face_clusters WHERE id = ?", (cid,))
                deleted_clusters += 1

                # Изчистваме директорията на cluster-а
                cluster_dir = FACES_DIR / str(cid)
                try:
                    shutil.rmtree(cluster_dir, ignore_errors=True)
                except Exception:
                    pass

                # Чистим reference-ите в detection_events.faces JSON
                # (face полето е list of {cluster_id, name})
                conn.execute(
                    "UPDATE detection_events SET faces = NULL "
                    "WHERE faces LIKE ?",
                    (f'%"cluster_id": {cid}%',),
                )

            return deleted_clusters, deleted_crops, deleted_files

    cl, cr, fl = await asyncio.to_thread(_bulk)
    _LOGGER.info("Bulk faces delete: clusters=%d crops=%d files=%d keep_named=%s ch=%s",
                 cl, cr, fl, keep_named, channel)
    return {"deleted_clusters": cl, "deleted_crops": cr, "deleted_files": fl}


@app.delete("/api/recordings/bulk")
async def recordings_bulk_delete(
    older_than_days: int        = Query(..., ge=0),
    channel:         int | None = None,
):
    """Bulk delete на MP4 recording segments.

    Query params:
      ?older_than_days=N (REQUIRED, ≥0) — трие segments по-стари от N дни
        Note: 0 = трие ВСИЧКО (днес и преди)
      ?channel=CH                       — само от канал CH (default: всички)
    """
    base = Path(recording_settings.path)
    if not base.exists():
        return {"deleted_files": 0, "freed_bytes": 0}

    # Date-folder cutoff: YYYYMMDD по local time (recording paths са strftime-base
    # от TZ=Europe/Sofia). По-бързо от full rglob — оптимизация за огромни volumes.
    from datetime import datetime as _dt
    cutoff_local = _dt.now() - timedelta(days=older_than_days)
    cutoff_yyyymmdd = cutoff_local.strftime("%Y%m%d")
    deleted = 0
    freed   = 0

    def _bulk():
        nonlocal deleted, freed
        if channel is not None:
            cam_dirs = [base / str(channel)]
        else:
            cam_dirs = [d for d in base.iterdir() if d.is_dir() and d.name.isdigit()]

        for cam_dir in cam_dirs:
            if not cam_dir.exists():
                continue
            # Скенирай САМО date dirs (YYYYMMDD), не "live"
            for day_dir in list(cam_dir.iterdir()):
                if not day_dir.is_dir() or len(day_dir.name) != 8:
                    continue
                if not day_dir.name.isdigit():
                    continue
                # Триене ако date-folder name e <= cutoff
                if day_dir.name >= cutoff_yyyymmdd:
                    continue
                # Триене на цялата day-folder за efficiency
                try:
                    for f in day_dir.rglob("*.mp4"):
                        try:
                            freed += f.stat().st_size
                            deleted += 1
                        except Exception:
                            pass
                    shutil.rmtree(day_dir, ignore_errors=True)
                except Exception as exc:
                    _LOGGER.warning("bulk delete day=%s: %s", day_dir, exc)

    await asyncio.to_thread(_bulk)
    _LOGGER.info("Bulk recordings delete: files=%d freed=%dMB days=%d ch=%s",
                 deleted, freed // (1024 * 1024), older_than_days, channel)
    return {
        "deleted_files":    deleted,
        "freed_bytes":      freed,
        "freed_mb":         freed // (1024 * 1024),
    }


@app.get("/api/camera/detect/config")
async def camera_detect_config():
    """Arm статус + per-camera min_confidence + общ брой events."""
    nvr_channels    = _nvr_valid_channels()
    extra_channels  = [cam["ch"] for cam in _extra_cams_list()]
    shelly_channels = [cam["ch"] for cam in _shelly_cams_list()]
    all_channels    = nvr_channels + extra_channels + shelly_channels

    def _count():
        with sqlite3.connect(DB_PATH) as conn:
            return conn.execute("SELECT COUNT(*) FROM detection_events").fetchone()[0]

    total = await asyncio.to_thread(_count)
    return {
        "cameras": [
            {
                "channel":        ch,
                "armed":          _detect_armed.get(ch, True),
                "min_confidence": _detect_conf.get(ch),
            }
            for ch in all_channels
        ],
        "total_events":      total,
        "max_events":        1000,
        "global_confidence": detect_settings.confidence,
    }


@app.post("/api/camera/{channel}/detect")
async def camera_detect_arm(channel: int, body: DetectArmBody):
    """ARM / DISARM детекция за конкретна камера (NVR + RTSP + Shelly)."""
    cam = _get_camera(channel)
    if not cam:
        raise HTTPException(404, "Camera not found")
    _detect_armed[channel] = body.armed
    await asyncio.to_thread(_db_set_armed, channel, body.armed)
    _LOGGER.info("Detection cam%d: %s", channel, "ARMED" if body.armed else "DISARMED")
    return {"channel": channel, "armed": body.armed}





class DetectArmBody(BaseModel):
    armed: bool


class DetectConfBody(BaseModel):
    min_confidence: float | None   # None → нулира към глобалната стойност


@app.post("/api/camera/{channel}/detect/confidence")
async def camera_detect_confidence(channel: int, body: DetectConfBody):
    """Задава per-camera минимална confidence. null = ползва глобалната."""
    if not _get_camera(channel):
        raise HTTPException(404, "Camera not found")
    mc = body.min_confidence
    if mc is not None:
        mc = max(0.1, min(0.99, mc))
    _detect_conf[channel] = mc
    await asyncio.to_thread(_db_set_min_confidence, channel, mc)
    _LOGGER.info("Detection cam%d: min_confidence=%s", channel, mc)
    return {"channel": channel, "min_confidence": mc}


# COCO клас → Bulgarian label (всички 80 класа)
_COCO_BG: dict[int, str] = {
    0:"човек", 1:"велосипед", 2:"кола", 3:"мотор", 4:"самолет",
    5:"автобус", 6:"влак", 7:"камион", 8:"лодка", 9:"светофар",
    10:"пожарникарски кран", 11:"знак стоп", 12:"паркомет",
    13:"пейка", 14:"птица", 15:"котка", 16:"куче", 17:"кон",
    18:"овца", 19:"крава", 20:"слон", 21:"мечка", 22:"зебра",
    23:"жираф", 24:"раница", 25:"чадър", 26:"чанта", 27:"вратовръзка",
    28:"куфар", 29:"фризби", 30:"ски", 31:"сноуборд", 32:"спортна топка",
    33:"хвърчило", 34:"бейзболна бухалка", 35:"бейзболна ръкавица",
    36:"скейтборд", 37:"сърф", 38:"тенис ракета", 39:"бутилка",
    40:"чаша вино", 41:"чаша", 42:"вилица", 43:"нож", 44:"лъжица",
    45:"купа", 46:"банан", 47:"ябълка", 48:"сандвич", 49:"портокал",
    50:"броколи", 51:"морков", 52:"хот-дог", 53:"пица", 54:"поничка",
    55:"торта", 56:"стол", 57:"диван", 58:"саксия", 59:"легло",
    60:"маса", 61:"тоалетна", 62:"телевизор", 63:"лаптоп",
    64:"мишка", 65:"дистанционно", 66:"клавиатура", 67:"телефон",
    68:"микровълнова", 69:"фурна", 70:"тостер", 71:"мивка",
    72:"хладилник", 73:"книга", 74:"часовник", 75:"ваза",
    76:"ножица", 77:"плюшена играчка", 78:"сешоар", 79:"четка за зъби",
}

# Количествено число → граматична форма (нула/ед.ч/мн.ч)
def _bg_count(n: int, word_singular: str, word_plural: str) -> str:
    if n == 1:
        return f"1 {word_singular}"
    return f"{n} {word_plural}"

# Множествена форма за COCO labels (опростена)
_COCO_BG_PLURAL: dict[int, str] = {
    0:"човека", 2:"коли", 3:"мотора", 5:"автобуса", 7:"камиона",
    15:"котки", 16:"кучета", 24:"раници", 56:"стола", 60:"маси",
    62:"телевизора", 63:"лаптопа",
}


@app.get("/api/camera/{channel}/describe")
async def camera_describe(channel: int, hq: bool = False):
    """
    On-demand YOLO описание на текущия кадър от камерата.
    Връща Bulgarian текст с всички засечени обекти (80 COCO класа).
    ?hq=1 → използва main stream snapshot за по-добро качество.

    Поддържа и NVR Dahua канали, и Extra IP камери (Reolink/IMOU).
    """
    if channel not in _all_valid_channels():
        raise HTTPException(404, "Камерата не е намерена")

    extra = _extra_cam(channel)
    if not extra and not nvr_settings.host:
        raise HTTPException(404, "Камерата не е намерена")

    loop = asyncio.get_running_loop()
    jpeg: bytes | None = None

    # Определяме RTSP URL според типа камера
    if extra:
        rtsp = extra["rtsp_main"] if hq else (extra.get("rtsp_sub") or extra["rtsp_main"])
    else:
        user, pw, host = nvr_settings.user, nvr_settings.password, nvr_settings.host
        subtype = 0 if hq else 1
        rtsp = f"rtsp://{user}:{pw}@{host}:554/cam/realmonitor?channel={channel}&subtype={subtype}"

    if hq or extra:
        # HQ snapshot или extra cam → винаги RTSP grab (нямаме HTTP snapshot за extra)
        cmd = [
            "ffmpeg", "-y",
            "-rtsp_transport", "tcp",
            "-analyzeduration", "5000000", "-probesize", "5000000",
            "-i", rtsp, "-an",
            "-vf", "select=eq(pict_type\\,I)",
            "-vsync", "vfr",
            "-frames:v", "1", "-q:v", "2",
            "-f", "image2", "pipe:1",
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=12)
            if stdout and len(stdout) > 5000:
                jpeg = stdout
        except Exception:
            pass

    if not jpeg and not extra:
        # NVR fallback: cached snapshot или HTTP
        cached = _nvr_snap_cache.get(channel)
        if cached:
            jpeg = cached[1]
        else:
            url  = f"http://{nvr_settings.host}/cgi-bin/snapshot.cgi?channel={channel}"
            auth = httpx.DigestAuth(nvr_settings.user, nvr_settings.password)
            async with httpx.AsyncClient(timeout=10.0) as cl:
                r = await cl.get(url, auth=auth)
                if r.status_code == 200:
                    jpeg = r.content
    elif not jpeg and extra:
        # Extra cam: cached snapshot fallback
        cached = _nvr_snap_cache.get(channel)
        if cached:
            jpeg = cached[1]

    if not jpeg:
        raise HTTPException(502, "Не може да се вземе снимка от камерата")

    # YOLO inference (всички 80 класа, по-нисък праг за описание)
    def _run_yolo(data: bytes) -> list[dict]:
        import io as _io, numpy as _np
        from PIL import Image as _Pil
        from ultralytics import YOLO as _YOLO

        model = _YOLO(detect_settings.model)
        img   = _Pil.open(_io.BytesIO(data)).convert("RGB")
        res   = model.predict(_np.array(img), conf=0.25, verbose=False, save=False)
        found = []
        for r in res:
            for box in r.boxes:
                found.append({
                    "class_id":  int(box.cls[0]),
                    "label":     _COCO_BG.get(int(box.cls[0]), str(int(box.cls[0]))),
                    "conf":      round(float(box.conf[0]), 2),
                })
        return found

    detections = await loop.run_in_executor(None, _run_yolo, jpeg)

    if not detections:
        description = "Нищо конкретно не се вижда в кадъра."
    else:
        # Групираме по клас и броим
        from collections import Counter
        counts = Counter(d["label"] for d in detections)
        parts  = []
        for label, n in counts.most_common():
            cls_id = next((d["class_id"] for d in detections if d["label"] == label), -1)
            plural = _COCO_BG_PLURAL.get(cls_id, label + ("а" if n > 1 else ""))
            parts.append(_bg_count(n, label, plural))
        description = "Виждам: " + ", ".join(parts) + "."

    return {
        "channel":     channel,
        "description": description,
        "detections":  detections,
        "total":       len(detections),
    }


# ── Shelly Camera — виртуален канал 99 ───────────────────────────────────
SHELLY_CH = 99   # запазен за /api/shelly-cam/ingest endpoint (legacy)

# DEFAULT seed списъци — използват се САМО при първо стартиране (празна
# `cameras` таблица), за continuity с предишната hardcoded конфигурация.
# Runtime четене → винаги през DB cache (`_get_camera()`, `_shelly_cams_list()`,
# `_extra_cams_list()`).
SHELLY_CAMS_CFG_DEFAULT = [
    {"ch": 99,  "ip": "192.168.3.227", "name": "Shelly Cam 1"},
    {"ch": 100, "ip": "192.168.3.201", "name": "Shelly Cam 2"},
    {"ch": 101, "ip": "192.168.3.108", "name": "Shelly Cam 3"},
]
EXTRA_CAMS_CFG_DEFAULT = [
    {
        "ch":         7,
        "ip":         "192.168.3.120",
        "name":       "Patio-Pool",
        "user":       "admin",
        "password":   "monika20",
        "snap_url":   "http://192.168.3.120:8080/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=hh&user=admin&password=monika20",
        "rtsp_main":  "rtsp://admin:monika20@192.168.3.120:554/h264Preview_01_main",
        "rtsp_sub":   "rtsp://admin:monika20@192.168.3.120:554/h264Preview_01_sub",
        "has_audio":  True,
    },
]
_shelly_last_event: float = 0.0   # за /api/shelly-cam/ingest endpoint


def _shelly_cams_list() -> list[dict]:
    """Връща списък със Shelly камери от DB cache (само enabled).
    Output shape е като SHELLY_CAMS_CFG_DEFAULT за backward compat."""
    out: list[dict] = []
    for ch, cam in _cameras_get_all().items():
        if cam["type"] != "shelly" or not cam["enabled"]:
            continue
        out.append({
            "ch":   cam["channel"],
            "ip":   cam["config"].get("ip", ""),
            "name": cam["name"],
        })
    return sorted(out, key=lambda c: c["ch"])


def _extra_cams_list() -> list[dict]:
    """Връща списък с RTSP/Reolink камери от DB cache (само enabled).
    Output shape като EXTRA_CAMS_CFG_DEFAULT."""
    out: list[dict] = []
    for ch, cam in _cameras_get_all().items():
        if cam["type"] != "rtsp_generic" or not cam["enabled"]:
            continue
        cfg = cam["config"]
        out.append({
            "ch":         cam["channel"],
            "ip":         cfg.get("ip", ""),
            "name":       cam["name"],
            "user":       cfg.get("user", ""),
            "password":   cfg.get("password", ""),
            "snap_url":   cfg.get("snap_url", ""),
            "rtsp_main":  cfg.get("rtsp_main", ""),
            "rtsp_sub":   cfg.get("rtsp_sub", ""),
            "has_audio":  cam["has_audio"],
        })
    return sorted(out, key=lambda c: c["ch"])


def _extra_cam(ch: int) -> dict | None:
    """Връща config dict за extra IP камера от DB cache, или None."""
    cam = _get_camera(ch)
    if not cam or cam["type"] != "rtsp_generic" or not cam["enabled"]:
        return None
    cfg = cam["config"]
    return {
        "ch":         cam["channel"],
        "ip":         cfg.get("ip", ""),
        "name":       cam["name"],
        "user":       cfg.get("user", ""),
        "password":   cfg.get("password", ""),
        "snap_url":   cfg.get("snap_url", ""),
        "rtsp_main":  cfg.get("rtsp_main", ""),
        "rtsp_sub":   cfg.get("rtsp_sub", ""),
        "has_audio":  cam["has_audio"],
    }


@app.get("/api/shelly-cam/{channel}/video")
async def shelly_cam_video(channel: int, fps: int = 5):
    """
    MJPEG поток за Shelly камера — взима /camera/0/snapshot на FPS интервали.
    Сервира се като multipart/x-mixed-replace — директно в <img> тага.
    """
    cam = _get_camera(channel)
    if not cam or cam["type"] != "shelly":
        raise HTTPException(404, "Shelly camera not found")

    fps      = max(1, min(fps, 10))
    interval = 1.0 / fps
    boundary = "shellymjpeg"
    ip       = cam["config"].get("ip", "")
    if not ip:
        raise HTTPException(503, "Shelly camera IP not configured")

    async def generate():
        async with httpx.AsyncClient(timeout=httpx.Timeout(8.0)) as client:
            while True:
                t0 = time.monotonic()
                try:
                    resp = await client.get(f"http://{ip}/camera/0/snapshot")
                    if resp.status_code == 200 and len(resp.content) > 20_000:
                        frame  = resp.content
                        header = (
                            f"--{boundary}\r\n"
                            f"Content-Type: image/jpeg\r\n"
                            f"Content-Length: {len(frame)}\r\n\r\n"
                        ).encode()
                        yield header + frame + b"\r\n"
                except asyncio.CancelledError:
                    return
                except Exception:
                    pass
                elapsed = time.monotonic() - t0
                await asyncio.sleep(max(0.0, interval - elapsed))

    return StreamingResponse(
        generate(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/shelly-cam/ingest")
async def shelly_cam_ingest(request: Request):
    """
    Приема JPEG кадър (application/octet-stream) изпратен от браузъра,
    пуска YOLO detection и съхранява event при намерен обект.
    Throttle: минимум NVR_DETECT_COOLDOWN сек между 2 последователни event-а.
    Връща: {detections: [...], saved: bool}
    """
    global _shelly_last_event

    body = await request.body()
    if not body or len(body) < 1000:
        raise HTTPException(400, "Липсва JPEG тяло")

    detect_cls  = [int(x.strip()) for x in detect_settings.classes.split(",") if x.strip()]
    conf_thresh = detect_settings.confidence

    def _run(data: bytes) -> list[tuple[str, float]]:
        import io as _io, numpy as _np
        from PIL import Image as _Pil
        from ultralytics import YOLO as _YOLO
        img    = _Pil.open(_io.BytesIO(data)).convert("RGB")
        img_np = _np.array(img)
        model  = _YOLO(detect_settings.model)
        results = model.predict(img_np, classes=detect_cls, conf=conf_thresh,
                                verbose=False, save=False)
        found = []
        for r in results:
            for box in r.boxes:
                found.append((
                    _DETECT_LABELS.get(int(box.cls[0]), str(int(box.cls[0]))),
                    round(float(box.conf[0]), 3),
                ))
        return found

    loop       = asyncio.get_running_loop()
    detections = await loop.run_in_executor(None, _run, body)

    result = [{"label": d[0], "conf": d[1]} for d in detections]

    if not detections:
        return {"detections": result, "saved": False}

    now = time.monotonic()
    if now - _shelly_last_event < detect_settings.cooldown:
        return {"detections": result, "saved": False}
    _shelly_last_event = now

    detected_at   = datetime.now(timezone.utc)
    labels        = [d[0] for d in detections]
    confs         = [d[1] for d in detections]
    unique_labels = list(dict.fromkeys(labels))

    cam_dir  = EVENTS_DIR / str(SHELLY_CH)
    cam_dir.mkdir(parents=True, exist_ok=True)
    ts_str   = detected_at.strftime("%Y%m%d_%H%M%S")
    img_path = cam_dir / f"{ts_str}_{'_'.join(unique_labels)}.jpg"
    img_path.write_bytes(body)

    await asyncio.to_thread(
        _db_insert_detection,
        SHELLY_CH, detected_at.isoformat(), labels, confs, str(img_path),
    )
    _LOGGER.info("Shelly detection: %s", unique_labels)
    return {"detections": result, "saved": True}


@app.post("/api/shelly-cam/describe")
async def shelly_cam_describe(request: Request):
    """
    Приема JPEG (octet-stream) и връща Bulgarian описание на съдържанието.
    """
    body = await request.body()
    if not body or len(body) < 1000:
        raise HTTPException(400, "Липсва JPEG тяло")

    def _run_yolo(data: bytes) -> list[dict]:
        import io as _io, numpy as _np
        from PIL import Image as _Pil
        from ultralytics import YOLO as _YOLO
        model = _YOLO(detect_settings.model)
        img   = _Pil.open(_io.BytesIO(data)).convert("RGB")
        res   = model.predict(_np.array(img), conf=0.25, verbose=False, save=False)
        found = []
        for r in res:
            for box in r.boxes:
                found.append({
                    "class_id": int(box.cls[0]),
                    "label":    _COCO_BG.get(int(box.cls[0]), str(int(box.cls[0]))),
                    "conf":     round(float(box.conf[0]), 2),
                })
        return found

    loop       = asyncio.get_running_loop()
    detections = await loop.run_in_executor(None, _run_yolo, body)

    if not detections:
        description = "Нищо конкретно не се вижда в кадъра."
    else:
        from collections import Counter
        counts = Counter(d["label"] for d in detections)
        parts  = []
        for label, n in counts.most_common():
            cls_id = next((d["class_id"] for d in detections if d["label"] == label), -1)
            plural = _COCO_BG_PLURAL.get(cls_id, label + ("а" if n > 1 else ""))
            parts.append(_bg_count(n, label, plural))
        description = "Виждам: " + ", ".join(parts) + "."

    return {"channel": SHELLY_CH, "description": description, "detections": detections}


@app.get("/api/camera/{channel}/snapshot")
async def camera_snapshot(channel: int, live: bool = False, hq: bool = False):
    """
    Snapshot от NVR камера или Extra IP камера (Reolink/IMOU).
    - ?live=1  → bypass 12-секундния кеш (пресен кадър)
    - ?hq=1    → full-резолюция чрез RTSP + ffmpeg (кеш 30 с)
    """
    if channel not in _all_valid_channels():
        raise HTTPException(404, "Каналът не е намерен")

    extra = _extra_cam(channel)
    if not extra and not nvr_settings.host:
        raise HTTPException(503, "NVR не е конфигуриран")

    import io, time as _time
    now = _time.monotonic()

    # ── Extra IP камера ────────────────────────────────────────────────────
    if extra:
        cache_key = f"extra_{channel}_{'hq' if hq else 'sd'}"
        cache_s = 30 if hq else 8
        cached  = _nvr_snap_cache.get(cache_key)
        if cached and not live and (now - cached[0]) < cache_s:
            return StreamingResponse(io.BytesIO(cached[1]), media_type="image/jpeg")

        snap_url = extra.get("snap_url")
        stdout   = b""

        if snap_url:
            # Директен HTTP snapshot (Reolink API) — ~1.7s, full-res (2560×1920)
            try:
                async with httpx.AsyncClient(timeout=8.0) as _sc:
                    r = await _sc.get(snap_url)
                if r.status_code == 200 and len(r.content) > 5_000:
                    stdout = r.content
            except Exception as exc:
                _LOGGER.warning("Extra cam%d HTTP snap fail: %s", channel, exc)

        if not stdout:
            # Fallback: RTSP grab (ако snap_url липсва или е временно недостъпен)
            rtsp = extra["rtsp_main"] if hq else (extra.get("rtsp_sub") or extra["rtsp_main"])
            cmd = [
                "ffmpeg", "-y",
                "-rtsp_transport", "tcp",
                "-analyzeduration", "5000000", "-probesize", "5000000",
                "-i", rtsp, "-an",
                "-vf", "select=eq(pict_type\\,I)",
                "-vsync", "vfr",
                "-frames:v", "1",
                "-q:v", "2" if hq else "3",
                "-f", "image2", "pipe:1",
            ]
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
            except Exception as exc:
                raise HTTPException(502, f"ffmpeg грешка: {exc}")

        if not stdout or len(stdout) < 5_000:
            raise HTTPException(502, "Не може да се извлече snapshot от Extra камерата")
        _nvr_snap_cache[cache_key] = (now, stdout)
        return StreamingResponse(io.BytesIO(stdout), media_type="image/jpeg")

    # ── HQ режим: RTSP → ffmpeg → JPEG ──────────────────────────────
    if hq:
        HQ_CACHE_S = 30
        cached_hq  = _nvr_snap_cache.get(f"hq_{channel}")
        if cached_hq and (now - cached_hq[0]) < HQ_CACHE_S:
            return StreamingResponse(io.BytesIO(cached_hq[1]), media_type="image/jpeg")

        user = nvr_settings.user
        pw   = nvr_settings.password
        host = nvr_settings.host

        # Искаме от NVR да издаде I-frame СЕГА преди да отворим RTSP.
        _kf_auth2 = httpx.DigestAuth(user, pw)
        async with httpx.AsyncClient(timeout=4.0) as _kf_cl2:
            for _kf_url2 in [
                f"http://{host}/cgi-bin/video.cgi"
                f"?action=startKeyFrame&channel={channel}&streamType=main",
                f"http://{host}/cgi-bin/snapshot.cgi?channel={channel}",
            ]:
                try:
                    await _kf_cl2.get(_kf_url2, auth=_kf_auth2)
                    break
                except Exception:
                    continue
        await asyncio.sleep(0.35)

        rtsp = f"rtsp://{user}:{pw}@{host}:554/cam/realmonitor?channel={channel}&subtype=0"
        cmd  = [
            "ffmpeg", "-y",
            "-rtsp_transport", "tcp",
            "-analyzeduration", "5000000",
            "-probesize", "5000000",
            "-i", rtsp,
            "-an",
            "-vf", "select=eq(pict_type\\,I)",
            "-vsync", "vfr",
            "-frames:v", "1",
            "-q:v", "2",
            "-f", "image2",
            "pipe:1",
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        except asyncio.TimeoutError:
            raise HTTPException(504, "ffmpeg timeout")
        except Exception as exc:
            raise HTTPException(502, f"ffmpeg грешка: {exc}")

        if not stdout or proc.returncode != 0:
            # Fallback към стандартния snapshot при ffmpeg грешка
            _LOGGER.warning("Camera HQ ch%s: ffmpeg върна код %s, fallback", channel, proc.returncode)
        else:
            _nvr_snap_cache[f"hq_{channel}"] = (now, stdout)
            return StreamingResponse(io.BytesIO(stdout), media_type="image/jpeg")

    # ── Стандартен snapshot: HTTP CGI ────────────────────────────────
    cached = _nvr_snap_cache.get(channel)
    if not live and cached and (now - cached[0]) < nvr_settings.cache_s:
        return StreamingResponse(io.BytesIO(cached[1]), media_type="image/jpeg")

    url  = f"http://{nvr_settings.host}/cgi-bin/snapshot.cgi?channel={channel}"
    auth = httpx.DigestAuth(nvr_settings.user, nvr_settings.password)
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url, auth=auth)
            r.raise_for_status()
            data = r.content
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Грешка от NVR: {exc}")

    _nvr_snap_cache[channel] = (now, data)
    return StreamingResponse(io.BytesIO(data), media_type="image/jpeg")


@app.get("/api/camera/{channel}/stream")
async def camera_stream(channel: int):
    """Direct MJPEG passthrough от NVR HTTP CGI. Само за NVR Dahua канали;
    Extra IP камери (Reolink/IMOU) трябва да ползват /api/camera/{ch}/video."""
    if channel not in _nvr_valid_channels():
        if _extra_cam(channel):
            raise HTTPException(400, "Extra IP камерите ползват /video endpoint, не /stream")
        raise HTTPException(404, "Каналът не е намерен")
    if not nvr_settings.host:
        raise HTTPException(503, "NVR не е конфигуриран")

    url = f"http://{nvr_settings.host}/cgi-bin/mjpg/video.cgi?channel={channel}&subtype=1"
    auth = httpx.DigestAuth(nvr_settings.user, nvr_settings.password)

    client = httpx.AsyncClient(timeout=httpx.Timeout(connect=8.0, read=None, write=8.0, pool=8.0))
    stream_ctx = client.stream("GET", url, auth=auth)
    nvr_resp = await stream_ctx.__aenter__()

    if nvr_resp.status_code != 200:
        await stream_ctx.__aexit__(None, None, None)
        await client.aclose()
        raise HTTPException(502, f"NVR върна {nvr_resp.status_code}")

    content_type = nvr_resp.headers.get(
        "content-type", "multipart/x-mixed-replace; boundary=myboundary"
    )

    async def generate():
        try:
            async for chunk in nvr_resp.aiter_bytes(8192):
                yield chunk
        finally:
            await stream_ctx.__aexit__(None, None, None)
            await client.aclose()

    return StreamingResponse(generate(), media_type=content_type)


@app.get("/api/camera/{channel}/video")
async def camera_video(channel: int, fps: int = 10, hq: bool = False):
    """
    RTSP → MJPEG re-stream чрез ffmpeg с минимална латентност.
    Поддържа NVR Dahua канали и Extra IP камери (Reolink/IMOU).
    ?fps=N   → брой кадри/сек (1-25, default 10)
    ?hq=1    → main stream, default sub-stream
    Сервира се като multipart/x-mixed-replace — директно в <img> тага.
    """
    if channel not in _all_valid_channels():
        raise HTTPException(404, "Камерата не е намерена")

    extra = _extra_cam(channel)
    if not extra and not nvr_settings.host:
        raise HTTPException(404, "Камерата не е намерена")

    fps = max(1, min(fps, 25))

    if extra:
        # Extra IP камера — собствен RTSP URL (без NVR keyframe trigger)
        rtsp = extra["rtsp_main"] if hq else (extra.get("rtsp_sub") or extra["rtsp_main"])
    else:
        subtype = 0 if hq else 1
        user    = nvr_settings.user
        pw      = nvr_settings.password
        host    = nvr_settings.host
        rtsp    = f"rtsp://{user}:{pw}@{host}:554/cam/realmonitor?channel={channel}&subtype={subtype}"

        # Искаме от NVR да издаде I-frame СЕГА преди да отворим RTSP.
        _kf_auth = httpx.DigestAuth(user, pw)
        async with httpx.AsyncClient(timeout=4.0) as _kf_cl:
            for _kf_url in [
                f"http://{host}/cgi-bin/video.cgi"
                f"?action=startKeyFrame&channel={channel}&streamType=main",
                f"http://{host}/cgi-bin/snapshot.cgi?channel={channel}",
            ]:
                try:
                    await _kf_cl.get(_kf_url, auth=_kf_auth)
                    break
                except Exception:
                    continue
        await asyncio.sleep(0.35)

    # За H.265 HQ потоци НЕ използваме nobuffer — ffmpeg трябва да буферира
    # докато получи първия I-frame, иначе статичните зони остават сиви.
    # За sub-stream (H.264) nobuffer е ок — кратките GOP-ове не имат проблема.
    fflags_val = "genpts+discardcorrupt" if hq else "nobuffer+genpts+discardcorrupt"

    cmd = [
        "ffmpeg",
        "-fflags", fflags_val,
        "-rtsp_transport", "tcp",
        "-analyzeduration", "2000000",
        "-probesize", "2000000",
        "-i", rtsp,
        "-an",
        "-vf", f"fps={fps}",
        "-vsync", "drop",
        "-q:v", "4",
        "-f", "mjpeg",
        "-flush_packets", "1",
        "pipe:1",
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    boundary = "mjpegframe"
    # Максимален буфер: 2 пълни кадъра при HQ (≈600 KB) или 4 при sub (≈200 KB)
    MAX_BUF = 1_200_000 if hq else 800_000

    async def generate():
        buf = b""
        try:
            while True:
                chunk = await proc.stdout.read(65536)
                if not chunk:
                    break
                buf += chunk

                # Ако буферът е прекалено голям — skip стари кадри, вземи последния
                if len(buf) > MAX_BUF:
                    last = buf.rfind(b"\xff\xd8")
                    if last > 0:
                        buf = buf[last:]

                # Извличаме JPEG кадри (FF D8 … FF D9)
                while True:
                    s = buf.find(b"\xff\xd8")
                    if s == -1:
                        buf = b""
                        break
                    e = buf.find(b"\xff\xd9", s + 2)
                    if e == -1:
                        buf = buf[s:]
                        break
                    frame = buf[s : e + 2]
                    buf   = buf[e + 2:]
                    header = (
                        f"--{boundary}\r\n"
                        f"Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(frame)}\r\n\r\n"
                    ).encode()
                    yield header + frame + b"\r\n"
        except asyncio.CancelledError:
            pass
        finally:
            try:
                proc.kill()
            except Exception:
                pass
            await proc.wait()

    return StreamingResponse(
        generate(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.websocket("/api/camera/{channel}/ws")
async def camera_ws(websocket: WebSocket, channel: int):
    """MJPEG → WebSocket: извлича JPEG кадри от NVR потока и ги праща на клиента.
    Само NVR Dahua канали; Extra IP камери ползват /video endpoint."""
    if not nvr_settings.host or channel not in _nvr_valid_channels():
        await websocket.close(code=1008)
        return

    await websocket.accept()

    url  = f"http://{nvr_settings.host}/cgi-bin/mjpg/video.cgi?channel={channel}&subtype=1"
    auth = httpx.DigestAuth(nvr_settings.user, nvr_settings.password)

    client     = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=8.0))
    stream_ctx = client.stream("GET", url, auth=auth)

    try:
        nvr_resp = await stream_ctx.__aenter__()
        if nvr_resp.status_code != 200:
            await websocket.close(code=1011)
            return

        buf         = b""
        frame_count = 0
        _CL_RE      = re.compile(rb"Content-Length:\s*(\d+)", re.IGNORECASE)

        async for chunk in nvr_resp.aiter_bytes(16384):
            buf += chunk
            # Парсинг по Content-Length — по-надежден от FF D8/D9
            while True:
                # Намираме края на multipart хедъра (\r\n\r\n)
                hdr_end = buf.find(b"\r\n\r\n")
                if hdr_end == -1:
                    break
                header_section = buf[:hdr_end]
                m = _CL_RE.search(header_section)
                if not m:
                    # Няма Content-Length — fallback: търсим FF D8
                    s = buf.find(b"\xff\xd8")
                    if s == -1:
                        buf = b""
                        break
                    e = buf.find(b"\xff\xd9", s + 2)
                    if e == -1:
                        buf = buf[s:]
                        break
                    frame = buf[s : e + 2]
                    buf   = buf[e + 2:]
                else:
                    cl    = int(m.group(1))
                    start = hdr_end + 4
                    if len(buf) < start + cl:
                        break          # нямаме всички байтове още
                    frame = buf[start : start + cl]
                    buf   = buf[start + cl:]

                if len(frame) < 100:   # пропусни прекалено малки "кадри"
                    continue
                await websocket.send_bytes(frame)
                frame_count += 1
                if frame_count == 1:
                    _LOGGER.info("Camera WS ch%s: първи кадър %d bytes", channel, len(frame))

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        _LOGGER.warning("Camera WS ch%s грешка: %s", channel, exc)
    finally:
        await stream_ctx.__aexit__(None, None, None)
        await client.aclose()


# ── Face Recognition API ────────────────────────────────────────────────────

@app.get("/api/faces/clusters")
async def faces_list_clusters():
    """Всички клъстери с брой лица и representative image URL."""
    clusters = await asyncio.to_thread(_db_list_clusters)
    result = []
    for c in clusters:
        rep_url = None
        if c["representative"] and Path(c["representative"]).exists():
            rep_url = f"/api/faces/crop-image?path={c['representative']}"
        result.append({
            "id":           c["id"],
            "name":         c["name"],
            "face_count":   c["face_count"],
            "created_at":   c["created_at"],
            "representative": rep_url,
        })
    return result


@app.get("/api/faces/clusters/{cluster_id}/crops")
async def faces_cluster_crops(cluster_id: int):
    """Всички face crops за даден клъстер."""
    crops = await asyncio.to_thread(_db_get_cluster_crops, cluster_id)
    result = []
    for c in crops:
        result.append({
            "id":           c["id"],
            "detected_at":  c["detected_at"],
            "channel":      c["channel"],
            "image_url":    f"/api/faces/crop-image?path={c['image_path']}",
        })
    return result


@app.get("/api/faces/crop-image")
async def faces_crop_image(path: str):
    """Сервира файл с face crop по абсолютен path (само от FACES_DIR)."""
    p = Path(path).resolve()
    if not str(p).startswith(str(FACES_DIR.resolve())):
        raise HTTPException(403, "Forbidden")
    if not p.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(str(p), media_type="image/jpeg")


class FaceNameRequest(BaseModel):
    name: str


@app.post("/api/faces/clusters/{cluster_id}/name")
async def faces_name_cluster(cluster_id: int, body: FaceNameRequest):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "Името е задължително")
    await asyncio.to_thread(_db_name_cluster, cluster_id, name)
    # Обнови всички detection_events.faces, които съдържат този cluster_id
    updated = await asyncio.to_thread(_db_propagate_cluster_name, cluster_id, name)
    return {"ok": True, "updated_events": updated}


class FaceMergeRequest(BaseModel):
    src_id: int
    dst_id: int


@app.post("/api/faces/clusters/merge")
async def faces_merge_clusters(body: FaceMergeRequest):
    if body.src_id == body.dst_id:
        raise HTTPException(400, "Не може да обедините клъстер сам със себе си")
    await asyncio.to_thread(_db_merge_clusters, body.src_id, body.dst_id)
    return {"ok": True}


@app.delete("/api/faces/clusters/{cluster_id}")
async def faces_delete_cluster(cluster_id: int):
    paths = await asyncio.to_thread(_db_delete_cluster, cluster_id)
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass
    # Изтриваме и папката ако е празна
    cluster_dir = FACES_DIR / str(cluster_id)
    try:
        cluster_dir.rmdir()
    except Exception:
        pass
    return {"ok": True, "deleted_crops": len(paths)}


@app.get("/api/faces/stats")
async def faces_stats():
    """Обобщена статистика за лица."""
    with sqlite3.connect(DB_PATH) as conn:
        total_clusters = conn.execute("SELECT COUNT(*) FROM face_clusters").fetchone()[0]
        named_clusters = conn.execute(
            "SELECT COUNT(*) FROM face_clusters WHERE name IS NOT NULL"
        ).fetchone()[0]
        total_crops    = conn.execute("SELECT COUNT(*) FROM face_crops").fetchone()[0]
    return {
        "total_clusters": total_clusters,
        "named_clusters": named_clusters,
        "total_crops":    total_crops,
    }


# ── Static ─────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(INDEX_HTML, media_type="text/html; charset=utf-8")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
