from os import PathLike
from typing import List, Union
from urllib.parse import parse_qs, urlparse

from torrentool.api import Torrent

from app.core.context import TorrentInfo
from app.log import logger
from app.plugins.brush4seed.config import brush4seed_config

# tracker的消息为这些时，认为不正常
tracker_invalid_msgs = [
    "not registered",
    "not exists",
    "unauthorized",
    "require passkey",
    "require authkey",
    "invalid passkey",
    "invalid authkey",
    "种子不存在",
    "该种子没有",
    "种子已被删除",
]


class TorrentService:

    @classmethod
    def get_tag_list(cls, torrent) -> List[str]:
        return torrent.tags.split(", ")

    @classmethod
    def get_first_tracker(cls, torrent) -> str:
        return torrent.tracker

    @classmethod
    def get_site_tag(cls, torrent):
        for tag in cls.get_tag_list(torrent):
            if tag.startswith(brush4seed_config.site_tag_prefix):
                return tag
        return None

    @classmethod
    def get_site_type(cls, torrent):
        site_tag = cls.get_site_tag(torrent)
        if site_tag is None:
            return None
        return site_tag.replace(brush4seed_config.site_tag_prefix, "")

    @classmethod
    def get_free_end_time(cls, torrent):
        torrent.name
        return None

    def check_trackers_validity(cls, torrent_trackers):
        # 检查tracker的msg，判断做种是否有效，返回false代表无效，应该删除种子
        has_invalid = False
        for tracker in torrent_trackers:
            # https://github.com/qbittorrent/qBittorrent/wiki/WebUI-API-(qBittorrent-4.1)#get-torrent-trackers
            if tracker.status == 0:
                continue
            if tracker.status == 2:
                return True
            if any(invalid_msg in tracker.msg for invalid_msg in tracker_invalid_msgs):
                has_invalid = True
                continue
        return False if has_invalid else True


def get_infohash_v1(torrent_file_path: Union[str, PathLike]) -> str:
    return Torrent.from_file(torrent_file_path).info_hash


def get_infohash_by_magnet_uri(magnet_uri: str):
    try:
        return parse_qs(urlparse(magnet_uri).query)["xt"][0].replace("urn:btih:", "")
    except Exception:
        logger.error(f"error when get infohash from {magnet_uri}", exc_info=True)
        return None


def get_total_size(torrent_list: List[TorrentInfo]) -> int:
    # 获取种子列表的总体积，单位byte
    return int(sum([torrent.size for torrent in torrent_list]))
