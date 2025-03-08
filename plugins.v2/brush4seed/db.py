import threading
from typing import Any, Optional

from app.db.plugindata_oper import PluginDataOper

plugin_data_oper = PluginDataOper()


class PluginDB:
    @classmethod
    def get_plugin_data(cls, key: str):
        return plugin_data_oper.get_data(cls.__class__.__name__, key)

    @classmethod
    def save_plugin_data(cls, key: str, value: Any):
        plugin_data_oper.save(cls.__class__.__name__, key, value)


class PluginDao:
    lock = threading.Lock()

    @classmethod
    def get_free_end_time(cls, torrent_hash) -> Optional[int]:
        db_torrents = PluginDB.get_plugin_data("torrents") or {}
        torrent_info = db_torrents.get(torrent_hash)
        if not torrent_info:
            return None
        free_end_time = torrent_info.get("free_end_time")
        assert isinstance(free_end_time, int)
        return free_end_time

    @classmethod
    def save_free_end_time(cls, torrent_hash, free_end_time):
        with cls.lock:
            db_torrents = PluginDB.get_plugin_data("torrents") or {}
            torrent_data = db_torrents.get(torrent_hash) or {}
            torrent_data["free_end_time"] = free_end_time
            PluginDB.save_plugin_data("torrents", db_torrents)
