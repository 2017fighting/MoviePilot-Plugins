import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, List, Optional

import pytz

from app.core.config import settings
from app.db.plugindata_oper import PluginDataOper


@dataclass
class TorrentSnapShot:
    hash: str
    state: str
    completed: str
    downloaded: str
    size: str
    total_size: str
    uploaded: str


@dataclass
class TorrentLog:
    site_type: str
    torrent_name: Optional[str]
    infohash: str
    log_datetime: str
    description: str

    @classmethod
    def new_log(
        cls,
        site_type,
        infohash,
        description=None,
        torrent_name=None,
    ):
        return cls(
            site_type=site_type,
            torrent_name=torrent_name,
            infohash=infohash,
            log_datetime=datetime.now(tz=pytz.timezone(settings.TZ)).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            description=description,
        )


class DataService:
    PLUGIN_ID = "Brush4Seed"
    _TORRENT_LOG_KEY = "torrents_log"
    _SNAPSHOT_PREFIX = "snapshot"
    lock = threading.Lock()

    @classmethod
    def _get_data(cls, key: str) -> Any:
        return PluginDataOper().get_data(cls.PLUGIN_ID, key)

    @classmethod
    def _save_data(cls, key: str, value: Any):
        PluginDataOper().save(cls.PLUGIN_ID, key, value)

    @classmethod
    def _del_data(cls, key: str):
        PluginDataOper().del_data(cls.PLUGIN_ID, key)

    @classmethod
    def _get_snapshot_key(cls, dt: datetime):
        date_str = dt.strftime("%Y%m%d")
        return f"{cls._SNAPSHOT_PREFIX}-{date_str}"

    @classmethod
    def save_snapshot(cls, torrents: List[TorrentSnapShot]):
        cls._save_data(
            cls._get_snapshot_key(datetime.now(tz=pytz.timezone(settings.TZ))),
            [asdict(t) for t in torrents],
        )

    @classmethod
    def get_snapshot(cls, dt: datetime) -> List[TorrentSnapShot]:
        data = cls._get_data(cls._get_snapshot_key(dt))
        if not data:
            return []
        return [TorrentSnapShot(**d) for d in data]

    @classmethod
    def get_torrent_logs(cls) -> List[TorrentLog]:
        data = cls._get_data(cls._TORRENT_LOG_KEY) or []
        return [TorrentLog(**d) for d in data]

    @classmethod
    def save_torrent_log(cls, torrent_log: List[TorrentLog]):
        with cls.lock:
            old_torrent_logs = cls.get_torrent_logs()
            # 由旧到新的顺序
            new_torrent_logs = old_torrent_logs + torrent_log
            # 只保留最新的500条记录
            new_torrent_logs = new_torrent_logs[-500:]
            # 存储到新的key上
            cls._save_data(
                cls._TORRENT_LOG_KEY, [asdict(log) for log in new_torrent_logs]
            )
