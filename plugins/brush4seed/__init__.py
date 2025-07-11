import functools
import itertools
import math
import random
import threading
import time
from bisect import bisect
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from itertools import accumulate
import copy
import shutil
from os import PathLike
from pathlib import Path
from types import NoneType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Set
from urllib.parse import parse_qs, urlparse

import pytz
from app import schemas
from app.chain.download import DownloadChain
from app.chain.torrents import TorrentsChain
from app.core.cache import cached
from app.core.config import settings
from app.core.context import TorrentInfo
from app.db.plugindata_oper import PluginDataOper
from app.db.site_oper import SiteOper
from app.helper.downloader import DownloaderHelper
from app.helper.sites import SitesHelper
from app.log import logger
from app.modules.qbittorrent.qbittorrent import Qbittorrent
from app.plugins import _PluginBase, PluginChian
from app.scheduler import Scheduler
from app.schemas import Notification, NotificationType, MessageChannel
from app.utils.singleton import Singleton
from app.utils.string import StringUtils
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from qbittorrentapi import (
    ApplicationPreferencesDictionary,
    TorrentCategoriesDictionary,
    TorrentDictionary,
    TorrentInfoList,
)
from torrentool.api import Torrent

####################################################error###################################################


class Brush4SeedBaseException(Exception):
    pass


class AddTorrentActionTypeError(Brush4SeedBaseException):
    pass


class NoEnoughDiskSpace(Brush4SeedBaseException):
    pass


class ExternalError(Brush4SeedBaseException):
    # 本插件依赖的外部服务出错
    pass


class InvalidDownloader(Brush4SeedBaseException):
    # 下载器出错
    pass


####################################################tag###################################################


class ErrorTag:
    INVALID_TRACKER = "error:unknown_tracker"
    FREE_END = "error:free_end"
    SLOW_DL = "error:slow_download"


####################################################util###################################################


def chunks(lst: List, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def choice_with_weight(population, weights):
    n = len(population)
    try:
        cum_weights = list(accumulate(weights))
    except TypeError:
        if not isinstance(weights, int):
            raise
        k = weights
        raise TypeError(
            f"The number of choices must be a keyword argument: {k=}"
        ) from None
    if len(cum_weights) != n:
        raise ValueError("The number of weights does not match the population")
    total = cum_weights[-1] + 0.0  # convert to float
    if total <= 0.0:
        raise ValueError("Total of weights must be greater than zero")
    if not math.isfinite(total):
        raise ValueError("Total of weights must be finite")
    hi = n - 1
    while True:
        yield population[bisect(cum_weights, random.random() * total, 0, hi)]


####################################################sort###################################################


# 为做种中的种子进行排序，考虑种子大小和做种人数(种子越大，做种人数越少，得分越多)
DEFAULT_EVAL_FOR_SEED = (
    """(t.size/ 1024.0/ 1024/ 1024)* (1 + math.sqrt(2) * math.pow(10, -1.0 * (p.seeds_total - 1) / 6))"""
)


def sort_key_for_seed(
    torrent: TorrentDictionary, eval_sort_score: str = DEFAULT_EVAL_FOR_SEED
):
    return eval(
        eval_sort_score,
        None,
        {"p": torrent.properties, "t": torrent},
    )


def sort_key_for_brush(torrent: TorrentInfo):
    try:
        # 最新发布的 如果只有一个人做种 经常会下载一两天也下载不完；这里确保不是只有一个人在做种
        if torrent.seeders <= 2:
            return 0
        dl_ul_rate = math.log(float(torrent.peers) / torrent.seeders + 1)
        final_score = 100 * dl_ul_rate * torrent.uploadvolumefactor
        # TODO: 最好考虑 种子发布时间 种子免费开始时间 的影响，但是目前MoviePilot对这两个字段的支持不太好
        return final_score
    except Exception:
        logger.error(
            f"站点:{torrent.site_name} 种子{torrent.site_name} 计算排序分数时出错;\n{torrent.seeders=} {torrent.peers=} {torrent.uploadvolumefactor=}"
        )
        return -1


####################################################tracker###################################################


class SiteConfig:
    site_type: str
    trackers: List[str]


class MTeamConfig(SiteConfig):
    site_type = "m-team.cc"
    trackers = [
        "tra1.m-team.cc",
    ]


class QingWaCofnig(SiteConfig):
    site_type = "qingwapt.com"
    trackers = [
        "tracker.qingwapt.org",
        "tracker.qingwa.pro",
        "tracker.qingwapt.com",
    ]


class SunnyConfig(SiteConfig):
    site_type = "sunnypt.top"
    trackers = [
        "sunnytk.top",
    ]


class AGSVConfig(SiteConfig):
    site_type = "agsvpt.com"
    trackers = [
        "agsvpt.trackers.work",
        "tracker.agsvpt.cn"
    ]


class SoulVoiceConfig(SiteConfig):
    site_type = "soulvoice.club"
    trackers = [
        "pt.soulvoice.club",
    ]


class AudiencesConfig(SiteConfig):
    site_type = "audiences.me"
    trackers = [
        "t.audiences.me",
        "tracker.cinefiles.info",
    ]


class LemonHDConfig(SiteConfig):
    site_type = "lemonhd.club"
    trackers = [
        "tracker01.ilovelemonhd.me",
    ]


class ZMConfig(SiteConfig):
    site_type = "zmpt.cc"
    trackers = [
        "zmpt.cc",
    ]


class PTTimeConfig(SiteConfig):
    site_type = "pttime"
    trackers = [
        "www.pttime.org",
    ]


class BTSchoolConfig(SiteConfig):
    site_type = "btschool.club"
    trackers = [
        "pt.btschool.club",
    ]


class HDKylinConfig(SiteConfig):
    site_type = "hdkyl.in"

    trackers = [
        "www.hdkylin.top",
        "tracker.hdkyl.in",
    ]


class RaingfhConfig(SiteConfig):
    site_type = "raingfh.top"
    trackers = [
        "raingfh.top",
    ]


class OnePTbaConfig(SiteConfig):
    site_type = "1ptba.com"
    trackers = [
        "1ptba.com",
    ]


class DMHYU2Config(SiteConfig):
    site_type = "u2.dmhy.org"
    trackers = [
        "daydream.dmhy.best",
    ]


class HTPTConfig(SiteConfig):
    site_type = "htpt.cc"
    trackers = [
        "www.htpt.cc",
    ]


class IloliconConfig(SiteConfig):
    site_type = "ilolicon.com"
    trackers = [
        "tracker.ilolicon.cc",
    ]


class MonikadesignConfig(SiteConfig):
    site_type = "monikadesign.uk"
    trackers = [
        "anime-no-index.com",
    ]


class HDDolbyConfig(SiteConfig):
    site_type = "hddolby.com"
    trackers = [
        "t.hddolby.com"
    ]


class SSDConfig(SiteConfig):
    site_type = "springsunday.net"
    trackers = [
        "on.springsunday.net"
    ]


class TrackerService:

    @classmethod
    def get_default_trackers(cls, site_type):
        for site_class in SiteConfig.__subclasses__():
            if site_class.site_type != site_type:
                continue
            return site_class.trackers
        return []


####################################################config###################################################


# "15/45 * * * 1-5"
DEFAULT_CRON_EXPR = "10 * * * *"
DEFAULT_CRON_EXPR_FOR_DELETE = "10 02 * * *"
Ti = 1099511627776
Gi = 1073741824


@dataclass
class Brush4SeedSiteConfig:
    brush_enable: bool = False
    min_dl_speed: Optional[int] = (
        None  # 下载速度最小值，平均下载速度低于这个值的会被暂停，单位bytes/s
    )
    target_seeding_size: float = 0  # 期望做种大小，单位TiB，未达到做种体积时会优先刷流
    trackers: List[str] = None  # 手动设置的tracker列表
    custom_sort_eval_for_seed: Optional[str] = (
        None  # 自定义该站点做种排序公式 例如某些站点做种不考虑种子大小
    )

    # 刷流种子过滤条件
    ensure_free: bool = True


@dataclass
class Brush4SeedConfig(metaclass=Singleton):
    managed_cate: str = "brush4seed"
    site_tag_prefix: str = "site:"
    no_delete_label: str = "no_delete"

    plugin_enable: bool = False
    run_now: bool = False

    delete_cron_enable: bool = False
    delete_run_now: bool = False
    confirm_delete: bool = False

    downloader_name: str = None
    cron_expr_list: List[str] = None
    cron_expr_list_for_delete: List[str] = None
    max_downloads: int = 3  # 同时下载任务数
    left_disk_space: float = 10  # 确保空闲空间大于这个值 单位TiB
    left_disk_space_after_delete:float = 15 #
    sites: Dict[str, Brush4SeedSiteConfig] = None  # 只有enable=true才加载进来

    @classmethod
    def _get_cron_list(cls, plugin_config:dict, cron_list_key, default_cron):
        cron_expr_str = plugin_config.get(cron_list_key)
        if not cron_expr_str:
            return [default_cron]
        else:
            return cron_expr_str.split("|")

    @classmethod
    def init_by_plugin_config(cls, plugin_config: dict):
        self = cls()
        self.plugin_enable = plugin_config.get("plugin_enable", False)
        self.run_now = plugin_config.get("run_now", False)
        self.delete_cron_enable = plugin_config.get("delete_cron_enable", False)
        self.delete_run_now = plugin_config.get("delete_run_now", False)
        self.confirm_delete = plugin_config.get("confirm_delete", False)
        self.downloader_name = plugin_config.get("downloader_name")
        self.cron_expr_list = cls._get_cron_list(plugin_config, "cron_expr_list", DEFAULT_CRON_EXPR)
        self.cron_expr_list_for_delete = cls._get_cron_list(plugin_config, "cron_expr_list_for_delete", DEFAULT_CRON_EXPR_FOR_DELETE)

        self.max_downloads = int(plugin_config.get("max_downloads", 3))
        self.left_disk_space = float(plugin_config.get("left_disk_space", 10))
        self.left_disk_space_after_delete = float(plugin_config.get("left_disk_space_after_delete", 15))

        tmp_sites = defaultdict(dict)
        # 因为前端的限制，必须是单层数据
        for site_key, site_value in plugin_config.items():
            if not site_key.startswith("sites"):
                continue
            site_key_sps = site_key.split("|")
            site_type = site_key_sps[1]
            site_config_key = site_key_sps[2]
            origin_value = site_value
            if site_config_key == "trackers":
                origin_value = site_value.split("\r\n")
            tmp_sites[site_type][site_config_key] = origin_value
        self.sites = {}

        for site in SitesHelper().get_indexers():
            site_type = StringUtils.get_url_domain(site.get("domain"))
            site_config = tmp_sites.get(site_type, {})
            self.sites[site_type] = Brush4SeedSiteConfig(
                brush_enable=site_config.get("brush_enable", False),
                min_dl_speed=int(site_config.get("min_dl_speed", 0)),
                target_seeding_size=float(site_config.get("target_seeding_size", 0)),
                trackers=site_config.get(
                    "trackers", TrackerService.get_default_trackers(site_type)
                ),
                custom_sort_eval_for_seed=site_config.get("custom_sort_eval_for_seed"),
                ensure_free=site_config.get("ensure_free", True),
            )
        return self

    def dump_to_plugin_config(self) -> dict:
        result = {}
        for k, v in asdict(self).items():
            # 由于前端限制，只能是单层的数据
            if k == "sites":
                for site_type, site_config in v.items():
                    for site_config_key, site_config_value in site_config.items():
                        final_key = f"sites|{site_type}|{site_config_key}"
                        if site_config_key == "trackers":
                            final_value = "\r\n".join(site_config_value or [])
                        elif site_config_key == "target_seeding_size":
                            final_value = 1.0 * site_config_value
                        else:
                            final_value = site_config_value
                        result[final_key] = final_value
            elif k in ["cron_expr_list", "cron_expr_list_for_delete"]:
                result[k] = "|".join(v)
            elif k in ["left_disk_space", "left_disk_space_after_delete"]:
                # 前端的单位是TiB
                result[k] = 1.0 * v
            else:
                result[k] = v
        return result

    def get_all_brush_enabled_sites(self) -> List[str]:
        return [
            site_type
            for site_type, site_config in self.sites.items()
            if site_config.brush_enable
        ]

    def get_site2target_seeding_size(self, only_brush_enabled=True) -> Dict[str, int]:
        site2target_seeding_size = defaultdict(int)
        for site_type, site_config in self.sites.items():
            if only_brush_enabled and site_config.brush_enable is False:
                continue
            site2target_seeding_size[site_type] = int(
                site_config.target_seeding_size * Ti
            )
        return site2target_seeding_size

    def get_left_disk_space_bytes(self):
        return int(self.left_disk_space * Ti)
    
    def get_left_disk_space_after_delete_in_bytes(self):
        return int(self.left_disk_space_after_delete * Ti)


####################################################site###################################################


class SiteService:
    def __init__(self, brush4seed_config):
        self.site_oper = SiteOper()
        self.torrents_chain = TorrentsChain()
        self.brush4seed_config = brush4seed_config

    def get_torrents_for_brush(self, site_type):
        site_info = self.site_oper.get_by_domain(site_type)
        if not site_info:
            logger.warning(f"站点{site_type}不存在")
            return []
        torrents = self.torrents_chain.browse(domain=site_type)
        if not torrents:
            logger.warning(f"站点{site_type}没有获取到种子")
            return []
        if not torrents:
            return []
        # 筛选
        site_config = self.get_site_config(self.brush4seed_config.sites, site_type)
        assert site_config is not None
        if site_config.ensure_free is True:
            torrents = [
                torrent for torrent in torrents if torrent.downloadvolumefactor == 0
            ]
        # TODO:其他筛选条件，例如hr、种子体积、种子名称
        # 排序
        for torrent in torrents:
            torrent.brush4seed_score = sort_key_for_brush(torrent)
        logger.info(f"站点:{site_type}\n{torrents=}")
        torrents.sort(key=lambda x: x.brush4seed_score, reverse=True)
        return torrents

    def get_eval_sort_score_for_seed(self, site_type: str):
        site_config = self.get_site_config(self.brush4seed_config.sites, site_type)
        if not site_config or site_config.custom_sort_eval_for_seed is None:
            return DEFAULT_EVAL_FOR_SEED
        return site_config.custom_sort_eval_for_seed

    @classmethod
    def get_site_config(
        cls, brush4seed_sites: Dict[str, Brush4SeedSiteConfig], site_type
    ) -> Optional[Brush4SeedSiteConfig]:
        return brush4seed_sites.get(site_type)

    @classmethod
    def get_site_type_by_tracker(
        cls, brush4seed_sites: Dict[str, Brush4SeedSiteConfig], tracker_url: str
    ) -> Optional[str]:
        netloc = urlparse(tracker_url).netloc
        for site_type, site_config in brush4seed_sites.items():
            if netloc in site_config.trackers:
                return site_type
        return None


####################################################torrent###################################################


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
    "该种子已被禁止",
    "无效passkey",
]


class TorrentService:

    @classmethod
    def get_tag_list(cls, torrent) -> List[str]:
        return torrent.tags.split(", ")

    @classmethod
    def get_first_tracker(cls, torrent) -> str:
        return torrent.tracker

    @classmethod
    def get_site_tag(cls, brush4seed_config, torrent):
        for tag in cls.get_tag_list(torrent):
            if tag.startswith(brush4seed_config.site_tag_prefix):
                return tag
        return None

    @classmethod
    def get_site_type(cls, brush4seed_config, torrent):
        site_tag = cls.get_site_tag(brush4seed_config, torrent)
        if site_tag is None:
            return None
        return site_tag.replace(brush4seed_config.site_tag_prefix, "")

    @classmethod
    def get_free_end_time(cls, torrent):
        torrent.name
        return None

    @classmethod
    def is_label_no_delete(cls, brush4seed_config, torrent):
        if brush4seed_config.no_delete_label in cls.get_tag_list(torrent):
            return True
        return False

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


####################################################download###################################################


class TorrentActionBase:
    pass


class EditTorrentActionBase:
    torrent_hash: str

    def __init__(self, torrent_hash: str):
        self.torrent_hash = torrent_hash


class AddTorrentAction(TorrentActionBase):
    magnet_uri: Optional[str] = None
    torrent_file_path: Optional[Path] = None

    # 新增其他字段时，需要修改函数DownloaderService._torrents_add中分组的逻辑
    category: Optional[str] = None
    tags: List[str] = []

    def __init__(
        self,
        magnet_uri: Optional[str] = None,
        torrent_file_path: Union[str, PathLike, NoneType] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        super().__init__()
        if magnet_uri is None:
            if torrent_file_path is None:
                raise AddTorrentActionTypeError(
                    "must specify one of magnet_uri and torrent_file_path"
                )
            else:
                torrent_file_path = Path(torrent_file_path)
                if not torrent_file_path.exists():
                    raise AddTorrentActionTypeError(
                        f"torrent file:{torrent_file_path} not exists"
                    )
                self.torrent_file_path = torrent_file_path
        elif torrent_file_path is not None:
            raise AddTorrentActionTypeError(
                "Cannot specify both magnet_uri and torrent_file_path"
            )
        else:
            self.magnet_uri = magnet_uri
        self.category = category
        self.tags = tags or []

    def __repr__(self):
        return f"AddTorrentAction({self.magnet_uri or self.torrent_file_path}, tags={self.tags}, cate={self.category})"


class TagTorrentAction(EditTorrentActionBase):
    tag: str

    def __init__(self, torrent_hash: str, tag: str):
        super().__init__(torrent_hash)
        self.tag = tag


class RemoveTagTorrentAction(EditTorrentActionBase):
    tags: List[str]

    def __init__(self, torrent_hash: str, tags: List[str]):
        super().__init__(torrent_hash)
        self.tags = tags


class DeleteTorrentAction(EditTorrentActionBase):
    delete_file: bool = True
    
    def __init__(self, torrent_hash, delete_file=True):
        super().__init__(torrent_hash)
        self.delete_file = delete_file


class StopTorrentAction(EditTorrentActionBase):
    torrent_hash: str
    pass


class DownloaderService:
    downloader: Qbittorrent

    def __init__(self, brush4seed_config: Brush4SeedConfig):
        self.brush4seed_config = brush4seed_config
        self.download_chain = DownloadChain()
        downloader_name = self.brush4seed_config.downloader_name
        downloader_service = DownloaderHelper().get_service(name=downloader_name)
        if not downloader_service:
            raise InvalidDownloader(f"下载器:{downloader_name},不存在")
        self.downloader = downloader_service.instance
        if not isinstance(self.downloader, Qbittorrent):
            raise InvalidDownloader("仅支持qBittorrent")
        if self.downloader.is_inactive():
            raise InvalidDownloader("下载器未连接")

    def get_preferences(self) -> ApplicationPreferencesDictionary:
        return self.downloader.qbc.app_preferences()

    def get_all_torrents(self) -> TorrentInfoList:
        return self.downloader.qbc.torrents_info()

    def get_seeding_torrents(self) -> TorrentInfoList:
        return self.downloader.qbc.torrents_info(
            status_filter="seeding"
        )
    
    def get_all_categories(self) -> TorrentCategoriesDictionary:
        return self.downloader.qbc.torrents_categories()

    def get_site2seeding_size(self) -> Dict[str, int]:
        site2seeding_size = defaultdict(int)
        for torrent in self.get_seeding_torrents():
            assert isinstance(torrent, TorrentDictionary)
            site_type = TorrentService.get_site_type(self.brush4seed_config, torrent)
            if site_type is None:
                continue
            site2seeding_size[site_type] += torrent.size
        return site2seeding_size

    def get_site2left_seeding_size(
        self, site2current_seeding_size=None
    ) -> Dict[str, int]:
        # 站点->该站点还差多少才能达到设置的保种体积(单位bytes)，仅返回相差值为正数的
        if site2current_seeding_size is None:
            site2current_seeding_size = self.get_site2seeding_size()
        site2target_seeding_size = self.brush4seed_config.get_site2target_seeding_size(
            only_brush_enabled=True
        )

        site2left_seeding_size = {}
        for site_type, target_seeding_size in site2target_seeding_size.items():
            current_seeding_size = site2current_seeding_size.get(site_type) or 0
            left_seeding_size = target_seeding_size - current_seeding_size
            if left_seeding_size <= 0:
                continue
            site2left_seeding_size[site_type] = left_seeding_size
        return site2left_seeding_size

    def get_site2over_seeding_size(
        self, site2current_seeding_size=None, only_over_seeding=True
    ) -> Dict[str, int]:
        # 站点->该站点比设置的保种体积多了多少(单位bytes)，仅返回多做种的站点
        if site2current_seeding_size is None:
            site2current_seeding_size = self.get_site2seeding_size()
        site2target_seeding_size = self.brush4seed_config.get_site2target_seeding_size(
            only_brush_enabled=False
        )
        site2over_seeding_size = {}
        for site_type, target_seeding_size in site2target_seeding_size.items():
            current_seeding_size = site2current_seeding_size.get(site_type) or 0
            over_seeding_size = current_seeding_size - target_seeding_size
            if only_over_seeding and over_seeding_size <= 0:
                continue
            site2over_seeding_size[site_type] = over_seeding_size
        return site2over_seeding_size

    def get_downloading_torrents(self) -> TorrentInfoList:
        return self.downloader.qbc.torrents_info(
            status_filter="downloading",
        )

    def bulk_execute_actions(self, torrent_actions: List[TorrentActionBase]):
        tag_actions = []
        remove_tag_actions = []
        delete_file2delete_actions = defaultdict(list)
        stop_actions = []
        add_actions = []
        hashes2delete = set()
        for action in torrent_actions:
            if isinstance(action, DeleteTorrentAction):
                hashes2delete.add(action.torrent_hash)
                delete_file2delete_actions[action.delete_file].append(action)
            elif isinstance(action, TagTorrentAction):
                if action.torrent_hash in hashes2delete:
                    continue
                tag_actions.append(action)
            elif isinstance(action, RemoveTagTorrentAction):
                remove_tag_actions.append(action)
            elif isinstance(action, StopTorrentAction):
                if action.torrent_hash in hashes2delete:
                    continue
                stop_actions.append(action)
            elif isinstance(action, AddTorrentAction):
                add_actions.append(action)
            else:
                logger.error(
                    f"invalid action type{type(action)} {id(action.__calss__)}"
                )
                raise NotImplementedError
        chunk_size = 100  # 每次最多同时操作100个种子
        for delete_file, delete_actions in delete_file2delete_actions.items():
            for delete_action_chunk in chunks(delete_actions, chunk_size):
                self._torrents_delete(delete_action_chunk, delete_file=delete_file)
        for tag_action_chunk in chunks(tag_actions, chunk_size):
            self._torrents_add_tags(tag_action_chunk)
        for remove_tag_action in remove_tag_actions:
            self._torrents_delete_tags(remove_tag_action)

        for stop_action_chunk in chunks(stop_actions, chunk_size):
            self._torrents_stop(stop_action_chunk)
        self._torrents_add(add_actions)

    def is_not_downloaded_before(self, torrent: TorrentInfo):
        # 确保之前没下载过
        assert torrent.enclosure
        download_result = self.download_torrent(torrent)
        # 获取infohash失败，认为没下载过
        if download_result["type"] == "failed":
            return False
        infohash = download_result["infohash"]
        torrent_infos = self.downloader.qbc.torrents_info(torrent_hashes=infohash)
        return True if not torrent_infos else False

    def download_torrent(self, torrent: TorrentInfo) -> Dict[str, str]:
        return self._download_torrent_and_cache(
            torrent.title,
            torrent.enclosure,
            torrent.site_cookie,
            torrent.site_ua,
            torrent.site_name,
            torrent.site_proxy,
        )

    @cached(region="brush4seed", ttl=259200)
    def _download_torrent_and_cache(
        self,
        torrent_title: str,
        torrent_enclosure: str,
        site_cookie: str,
        site_ua: str,
        site_name: str,
        site_proxy: bool,
    ) -> str:
        torrent = TorrentInfo(
            title=torrent_title,
            enclosure=torrent_enclosure,
            site_cookie=site_cookie,
            site_ua=site_ua,
            site_name=site_name,
            site_proxy=site_proxy,
        )
        torrent_file, download_folder, _ = self.download_chain.download_torrent(torrent)
        # 磁力链接
        if isinstance(torrent_file, str):
            return {
                "type": "magnet",
                "magnet_uri": torrent_file,
                "infohash": get_infohash_by_magnet_uri(torrent_file),
            }
        # 下载失败
        if not download_folder:
            return {"type": "failed"}
        infohash = get_infohash_v1(torrent_file)
        return {"type": "file", "file_path": torrent_file, "infohash": infohash}

    def get_free_space_on_disk(self) -> int:
        # 磁盘剩余空间，单位bytes
        return self.downloader.qbc.sync_maindata().server_state.free_space_on_disk

    def _torrents_add(self, torrent_add_actions: List[AddTorrentAction]):
        for group_key, group_actions in itertools.groupby(
            torrent_add_actions, key=lambda x: (x.category, x.tags)
        ):
            group_actions = list(group_actions)
            urls = [action.magnet_uri for action in group_actions if action.magnet_uri]
            torrent_files = [
                action.torrent_file_path
                for action in group_actions
                if action.torrent_file_path
            ]
            add_result = self.downloader.qbc.torrents_add(
                urls=urls,
                torrent_files=torrent_files,
                category=group_key[0],
                tags=group_key[1],
            )
            logger.info(f"add torrents[{urls},{torrent_files}] result:{add_result}")

    def _torrents_add_tags(self, torrent_tag_actions: List[TagTorrentAction]):
        tag2torrent_hashes = defaultdict(list)
        for tag_action in torrent_tag_actions:
            tag2torrent_hashes[tag_action.tag].append(tag_action.torrent_hash)
        for tag, torrent_hashes in tag2torrent_hashes.items():
            self.downloader.qbc.torrents_add_tags(
                tags=[tag], torrent_hashes=torrent_hashes
            )

    def _torrents_delete_tags(self, action: RemoveTagTorrentAction):
        self.downloader.qbc.torrents_remove_tags(
            tags=action.tags, torrent_hashes=action.torrent_hash
        )

    def _torrents_delete(self, torrent_delete_actions: List[DeleteTorrentAction], delete_file=True):
        self.downloader.qbc.torrents_delete(
            delete_files=delete_file,
            torrent_hashes=[action.torrent_hash for action in torrent_delete_actions],
        )

    def _torrents_stop(self, torrent_stop_actions: List[StopTorrentAction]):
        self.downloader.qbc.torrents_stop(
            [action.torrent_hash for action in torrent_stop_actions]
        )

    # def get_server_tags(self):
    #     return self.downloader.qbc.torrent_tags.tags


####################################################snapshot###################################################


class SnapShotService:

    @classmethod
    def _get_diff_uploaded(cls, brush4seed_config, infohash2last_snapshot, torrent):
        site_type = TorrentService.get_site_type(brush4seed_config, torrent)
        if not site_type:
            return None, 0
        last_uploaded = 0
        if torrent._torrent_hash in infohash2last_snapshot:
            last_uploaded = int(infohash2last_snapshot[torrent._torrent_hash].uploaded)
        diff_uploaded = torrent.uploaded - last_uploaded
        if diff_uploaded <= 0:
            diff_uploaded = 0
        return site_type, diff_uploaded

    @classmethod
    def snapshot(cls, brush4seed_config: Brush4SeedConfig = None):
        last_snapshot = DataService.get_snapshot()
        infohash2last_snapshot = {ss.hash: ss for ss in last_snapshot}
        site2diff_uploads = defaultdict(int)

        snapshots = []
        downloader_service = DownloaderService(brush4seed_config)
        all_torrents = downloader_service.get_all_torrents()
        for torrent in all_torrents:
            #计算差值，发送通知
            site_type, diff_uploaded = cls._get_diff_uploaded(brush4seed_config, infohash2last_snapshot, torrent)
            if site_type and diff_uploaded:
                site2diff_uploads[site_type] += diff_uploaded

            snapshots.append(
                TorrentSnapShot(
                    hash=torrent._torrent_hash,
                    state=torrent.state,
                    completed=str(torrent.completed),
                    downloaded=(torrent.downloaded),
                    size=str(torrent.size),
                    total_size=str(torrent.total_size),
                    uploaded=str(torrent.uploaded),
                ),
            )
        site_diff_uploads = [(site_type, diff_uploaded) for site_type, diff_uploaded in site2diff_uploads.items()]
        site_diff_uploads.sort(key=lambda x:x[1], reverse=True)
        NotifyService.post_message(
            mtype=NotificationType.SiteMessage,
            title="【刷流保种】每日上传快照",
            text="\n".join([
                f"{site_type} - {(diff_uploaded*1.0/Gi):.1f} GiB" for (site_type, diff_uploaded) in site_diff_uploads
            ])
        )
        DataService.save_snapshot(snapshots)


####################################################vuetify###################################################


class ComponentBase:
    component: Optional[str] = None
    props: Optional[dict] = None
    content: Optional[list] = None
    slots: Optional[dict] = None
    html: Optional[str] = None
    text: Optional[str] = None
    __fields__ = ["component", "props", "content", "slots", "html", "text"]

    def __init__(
        self,
        *content,
        slots: Optional[dict] = None,
        html: Optional[str] = None,
        text: Optional[str] = None,
        **props,
    ):
        self.content = list(content) or None
        self.slot = slots
        self.html = html
        self.text = text
        self.props = props or None

    def as_dict(self):
        result = {}
        for f in self.__fields__:
            value = getattr(self, f)
            if value is None:
                continue
            if f == "content":
                value = [c.as_dict() for c in value]
            result[f] = value
        return result

    def __repr__(self):
        return str(self.as_dict())


class VForm(ComponentBase):
    component: Optional[str] = "VForm"


class VRow(ComponentBase):
    component = "VRow"


class VCol(ComponentBase):
    component = "VCol"


class VSwitch(ComponentBase):
    component = "VSwitch"


class VSelect(ComponentBase):
    component = "VSelect"


class VBtn(ComponentBase):
    component = "VBtn"


class VTextField(ComponentBase):
    component = "VTextField"


class VCronField(ComponentBase):
    component = "VCronField"


class VTabs(ComponentBase):
    component = "VTabs"


class VTab(ComponentBase):
    component = "VTab"


class VWindow(ComponentBase):
    component = "VWindow"


class VWindowItem(ComponentBase):
    component = "VWindowItem"


class VBadge(ComponentBase):
    component = "VBadge"


class VTextarea(ComponentBase):
    component = "VTextarea"


class VApexChart(ComponentBase):
    component = "VApexChart"


class VDataTableVirtual(ComponentBase):
    component = "VDataTableVirtual"


####################################################form###################################################


class FormService:
    def __init__(self, brush4seed_config: Brush4SeedConfig):
        self.brush4seed_config = brush4seed_config

    def get_components(self) -> List[dict]:
        site_tabs, site_windows = self._get_site_tabs_and_windows()
        components = [
            VForm(
                VRow(
                    VCol(VSwitch(model="plugin_enable", label="刷流-定时运行")),
                    VCol(VSwitch(model="run_now", label="刷流-运行一次")),
                ),
                VRow(
                    VCol(VSwitch(model="delete_cron_enable", label="删种-定时运行")),
                    VCol(VSwitch(model="delete_run_now", label="删种-运行一次")),
                    VCol(VSwitch(model="confirm_delete", label="删种-确认删除")),
                ),
                VRow(
                    VCol(
                        VSelect(
                            required=True,
                            model="downloader_name",
                            label="下载器",
                            items=[
                                {"title": config.name, "value": config.name}
                                for config in DownloaderHelper().get_configs().values()
                            ],
                        )
                    ),
                    VCol(
                        VTextField(
                            required=True,
                            type="number",
                            model="max_downloads",
                            label="同时下载任务数",
                            min="0",
                        ),
                    ),
                    VCol(
                        VTextField(
                            required=True,
                            model="left_disk_space",
                            label="刷流时确保剩余磁盘空间大于，单位TiB",
                            type="number",
                            placeholder="10",
                        ),
                    ),
                    VCol(
                        VTextField(
                            required=True,
                            model="left_disk_space_after_delete",
                            label="删种后剩余磁盘空间大于，单位TiB",
                            type="number",
                            placeholder="15",
                        ),
                    ),
                ),
                VRow(
                    VCol(
                        VTextField(
                            model="cron_expr_list",
                            label="刷流执行周期",
                            placeholder="支持多个，竖线分隔，如果想工作日和假期不同频率，则类似于：15/45 * * * mon-fri|15 * * * sat,sun",
                        ),
                    )
                ),
                VRow(
                    VCol(
                        VTextField(
                            model="cron_expr_list_for_delete",
                            label="删种执行周期",
                            placeholder="支持多个，竖线分隔",
                        ),
                    )
                ),
                VRow(
                    VCol(
                        VTabs(
                            *site_tabs,
                            model="_site_tab",
                        )
                    ),
                ),
                VRow(VCol(VWindow(*site_windows, model="_site_tab"))),
                **{"validate-on": "blur"},
            ),
        ]
        return [c.as_dict() for c in components]

    def _get_site_tabs_and_windows(self):
        site_tabs = []
        site_windows = []
        for site in SitesHelper().get_indexers():
            site_type = StringUtils.get_url_domain(site.get("domain"))
            site_config = self.brush4seed_config.sites.get(site_type)
            brush_enable = False
            if site_config and site_config.brush_enable:
                brush_enable = True
            site_tabs.append(
                VTab(
                    VBadge(
                        # inline="true",
                        floating="true",
                        dot="true",
                        color="success" if brush_enable else "error",
                        text=site.get("name"),
                    ),
                    value=site_type,
                )
            )

            site_windows.append(
                VWindowItem(
                    VRow(
                        VCol(
                            VSwitch(
                                model=f"sites|{site_type}|brush_enable",
                                label="启用刷流",
                            )
                        ),
                        VCol(
                            VSwitch(
                                model=f"sites|{site_type}|ensure_free",
                                label="仅刷流免费种子",
                            )
                        ),
                    ),
                    VRow(
                        VCol(
                            VTextField(
                                model=f"sites|{site_type}|min_dl_speed",
                                label="下载速度最小值，单位bytes/s",
                                type="number",
                                placeholder="平均速度低于这个的会被暂停，设置为0代表关闭此项检查",
                            ),
                        ),
                        VCol(
                            VTextField(
                                model=f"sites|{site_type}|target_seeding_size",
                                label="期望做种大小，单位TiB",
                                type="number",
                                placeholder="未达到做种体积时会优先刷流",
                            ),
                        ),
                    ),
                    VRow(
                        VCol(
                            VTextarea(
                                model=f"sites|{site_type}|trackers",
                                label="trackers，用于判断种子属于哪个站点",
                            )
                        ),
                    ),
                    value=site_type,
                )
            )
        return site_tabs, site_windows

    def get_model(self) -> Dict[str, Any]:
        model = self.brush4seed_config.dump_to_plugin_config()
        # site_indexers = SitesHelper().get_indexers()
        # _site_tab = None
        # if site_indexers:
        #     _site_tab = StringUtils.get_url_domain(site_indexers[0].get("domain"))
        # model["_site_tab"] = _site_tab
        model["_site_tab"] = ""
        return model


####################################################data###################################################


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

class NotifyService:
    @classmethod
    def post_message(self, channel: MessageChannel = None, mtype: NotificationType = None, title: Optional[str] = None,
                 text: Optional[str] = None, image: Optional[str] = None, link: Optional[str] = None,
                 userid: Optional[str] = None, username: Optional[str] = None,
                 **kwargs):
        """
        发送消息
        """
        if not link:
            link = settings.MP_DOMAIN(f"#/plugins?tab=installed&id={Brush4Seed.__name__}")
        PluginChian().post_message(Notification(
            channel=channel, mtype=mtype, title=title, text=text,
            image=image, link=link, userid=userid, username=username, **kwargs
        ))

class DataService:
    PLUGIN_ID = "Brush4Seed"
    _TORRENT_LOG_KEY = "torrents_log"
    _TORRENTS_KEY = "torrents"
    _SNAPSHOT_KEY = "snapshot_today"
    _MAY_DELETE_FILE_LIST = "may_delete_file_list"
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
    def save_snapshot(cls, torrents: List[TorrentSnapShot]):
        cls._save_data(
            cls._SNAPSHOT_KEY,
            [asdict(t) for t in torrents],
        )

    @classmethod
    def get_snapshot(cls) -> List[TorrentSnapShot]:
        data = cls._get_data(cls._SNAPSHOT_KEY)
        if not data:
            return []
        return [TorrentSnapShot(**d) for d in data]

    @classmethod
    def get_torrent_logs(cls) -> List[TorrentLog]:
        data = cls._get_data(cls._TORRENT_LOG_KEY) or []
        return [TorrentLog(**d) for d in data]

    @classmethod
    def clear_torrent_logs(cls):
        cls._save_data(cls._TORRENT_LOG_KEY, [])

    @classmethod
    def save_torrent_log(cls, torrent_log: List[TorrentLog]):
        logger.info(f"save torrent log {torrent_log}")
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

    @classmethod
    def get_torrents(cls):
        return cls._get_data(cls._TORRENTS_KEY) or {}

    @classmethod
    def get_free_end_time(cls, torrent_hash) -> Optional[int]:
        db_torrents = cls.get_torrents()
        torrent_info = db_torrents.get(torrent_hash)
        if not torrent_info:
            return None
        free_end_time = torrent_info.get("free_end_time")
        assert isinstance(free_end_time, int)
        return free_end_time

    @classmethod
    def save_free_end_time(cls, torrent_hash, free_end_time):
        with cls.lock:
            db_torrents = cls.get_torrents()
            torrent_data = db_torrents.get(torrent_hash) or {}
            torrent_data["free_end_time"] = free_end_time
            cls._save_data("torrents", db_torrents)
    
    @classmethod
    def save_may_delete_file_list(cls, may_delete_file_list):
        cls._save_data(cls._MAY_DELETE_FILE_LIST, may_delete_file_list)
    
    @classmethod
    def get_may_delete_file_list(cls):
        return cls._get_data(cls._MAY_DELETE_FILE_LIST) or []

    @classmethod
    def clear_may_delete_file_list(cls):
        cls._save_data(cls._MAY_DELETE_FILE_LIST, [])


####################################################page###################################################


class PageService:
    def __init__(self, brush4seed_config: Brush4SeedConfig):
        self.plugin_data_oper = PluginDataOper()
        self.brush4seed_config = brush4seed_config

    def _get_pie_chart(self, title, labels, series, height=300, show_legend=True):
        return VApexChart(
            height=height,
            options={
                "chart": {
                    "type": "pie",
                },
                "labels": labels,
                "title": {
                    "text": title,
                },
                "legend": {
                    "show": show_legend,
                },
                "plotOptions": {
                    "pie": {"expandOnClick": False},
                },
                "noData": {"text": "暂无数据"},
            },
            series=series,
        )

    def _get_site_type2seeding_size(self, all_torrents):
        site_type2seeding_size = defaultdict(int)
        for torrent in all_torrents:
            if torrent.state not in ("stalledUP", "uploading"):
                continue

            site_type = TorrentService.get_site_type(self.brush4seed_config, torrent)
            if not site_type:
                continue
            site_type2seeding_size[site_type] += torrent.size
        return {
            site_type: 1.0 * seeding_size / Ti
            for site_type, seeding_size in site_type2seeding_size.items()
        }

    def _get_site2today_uploads(
        self,
        today_snapshot,
        all_torrents: TorrentInfoList,
    ):
        site2today_uploads = defaultdict(int)
        if not today_snapshot:
            return site2today_uploads
        infohash2snapshot = {ss.hash: ss for ss in today_snapshot}
        for torrent in all_torrents:
            site_type = TorrentService.get_site_type(self.brush4seed_config, torrent)
            if not site_type:
                continue
            torrent_snapshot = None
            today_uploads = torrent.uploaded
            if torrent._torrent_hash in infohash2snapshot:
                torrent_snapshot = infohash2snapshot[torrent._torrent_hash]
                assert isinstance(torrent_snapshot, TorrentSnapShot)
                today_uploads -= int(torrent_snapshot.uploaded)
            if today_uploads <= 0:
                continue
            site2today_uploads[site_type] += today_uploads
        return {
            site: 1.0 * today_uploads / Gi
            for site, today_uploads in site2today_uploads.items()
        }
    
    def _pretty_delete_log(self, may_delete_file):
        size_in_byte = may_delete_file["size_in_byte"]
        return {
            "path": may_delete_file["path"],
            "size_readable": "%.2f GiB" % (1.0*float(size_in_byte) / Gi),
            "seed_size": may_delete_file["seed_size"],
            "score_sum": "%.2f" % float(may_delete_file["score_sum"]),
        }

    def get_page(
        self,
        torrent_logs: List[TorrentLog],
        may_delete_file_list: List,
        all_torrents: TorrentInfoList,
        today_snapshot,
    ):
        site_type2seeding_TiB = self._get_site_type2seeding_size(all_torrents)
        seeding_sites = list(site_type2seeding_TiB.keys())
        seeding_total_TiB = 0
        seeding_sizes = []
        for site_type in seeding_sites:
            this_site_seeding_TiB = site_type2seeding_TiB[site_type]
            seeding_sizes.append(float(f"{this_site_seeding_TiB:.1f}"))
            seeding_total_TiB += this_site_seeding_TiB

        site2today_uploads = self._get_site2today_uploads(today_snapshot, all_torrents)
        upload_sites = list(site2today_uploads.keys())
        uploads = []
        upload_total_GiB = 0
        for site_type in upload_sites:
            this_site_upload_GiB = site2today_uploads[site_type]
            uploads.append(float(f"{this_site_upload_GiB:.1f}"))
            upload_total_GiB += this_site_upload_GiB

        page = [
            VRow(
                VCol(
                    self._get_pie_chart(
                        f"做种体积(TiB)，共{seeding_total_TiB:.2f}TiB",
                        seeding_sites,
                        seeding_sizes,
                    )
                ),
                VCol(
                    self._get_pie_chart(
                        f"今日上传(GiB)，共{upload_total_GiB:.2f}GiB",
                        upload_sites,
                        uploads,
                    )
                ),
            ),
            VRow(
                VCol(
                    text="刷流日志",
                )
            ),
            VRow(
                VCol(
                    VDataTableVirtual(
                        headers=[
                            {"title": "站点", "key": "site_type", "sortable": False},
                            {
                                "title": "种子名称",
                                "key": "torrent_name",
                                "sortable": False,
                            },
                            {
                                "title": "种子infohash",
                                "key": "infohash",
                                "sortable": False,
                            },
                            {"title": "时间", "key": "log_datetime", "sortable": False},
                            {"title": "描述", "key": "description", "sortable": False},
                        ],
                        items=list(reversed([asdict(log) for log in torrent_logs])),
                        height="30rem",
                        hover=True,
                        density="compact",
                        **{
                            "class": "text-sm",
                            "fixed-header": True,
                            "hide-no-data": True,
                        },
                    ),
                ),
            ),
            VRow(
                VCol(
                    text="删种日志",
                )
            ),
            VRow(
                VCol(
                    VDataTableVirtual(
                            headers=[
                                {"title":"文件路径", "key":"path", "sortable": False},
                                {"title":"文件大小", "key":"size_readable", "sortable": False},
                                {"title":"做种站点数", "key":"seed_size", "sortable": False},
                                {"title":"总评分", "key":"score_sum", "sortable": False},
                            ],
                            items=[self._pretty_delete_log(may_delete_file) for may_delete_file in may_delete_file_list],
                            height="30rem",
                            hover=True,
                            density="compact",
                            **{
                                "class": "text-sm",
                                "fixed-header": True,
                                "hide-no-data": True,
                            },
                        ),
                ),
            ),
        ]
        return [p.as_dict() for p in page]


####################################################brush###################################################


class TorrentsCheckerBase:
    def __init__(self, downloader_service: DownloaderService):
        self.downloader_service = downloader_service

    def _common_check(
        self,
        torrent_list,
        check_func: Callable[
            [TorrentDictionary], Tuple[List[TorrentActionBase], List[TorrentLog]]
        ],
    ) -> List[TorrentActionBase]:
        torrent_actions = []
        torrent_logs = []
        for torrent in torrent_list:
            assert isinstance(torrent, TorrentDictionary)
            if torrent._torrent_hash is None:
                logger.error("种子没有hash", torrent)
                continue
            actions, tlogs = check_func(torrent)
            if not actions:
                continue
            torrent_actions.extend(actions)
            torrent_logs.extend(tlogs)
        DataService.save_torrent_log(torrent_logs)
        return torrent_actions


class AllTorrentsChecker(TorrentsCheckerBase):

    def check(self, brush4seed_config):
        # 是否有site:xxx标签，没有的补全
        logger.info("check site tag...")
        all_torrents = self.downloader_service.get_all_torrents()
        self.downloader_service.bulk_execute_actions(
            self._common_check(
                all_torrents,
                functools.partial(
                    self._check_site_tag, brush4seed_config=brush4seed_config
                ),
            )
        )
        # tracker是否有效
        logger.info("check tracker...")
        self.downloader_service.bulk_execute_actions(
            self._common_check(all_torrents, self._check_tracker)
        )


    @staticmethod
    def _check_site_tag(torrent, brush4seed_config: Brush4SeedConfig = None):
        torrent_actions = []
        torrent_logs = []

        old_tag_list = TorrentService.get_tag_list(torrent)
        first_tracker = TorrentService.get_first_tracker(torrent)
        site_type = SiteService.get_site_type_by_tracker(
            brush4seed_config.sites, first_tracker
        )
        if site_type:
            new_tag = f"{brush4seed_config.site_tag_prefix}{site_type}"
        else:
            new_tag = ErrorTag.INVALID_TRACKER
            logger.info(f"torrent:{torrent._torrent_hash} unknown tracker:{first_tracker}, netloc: {urlparse(first_tracker).netloc}")
        if new_tag not in old_tag_list:
            torrent_actions.append(TagTorrentAction(torrent._torrent_hash, new_tag))
            torrent_logs.append(
                TorrentLog.new_log(
                    site_type or "-",
                    torrent._torrent_hash,
                    description=f"手动添加，新增标签{new_tag}",
                    torrent_name=torrent.name,
                )
            )
        # 删除之前添加的 site:xxx 和 ErrorTag.INVALID_TRACKER
        other_tags = set(old_tag_list) - set([new_tag])
        useless_tags = [
            tag
            for tag in other_tags
            if tag.startswith(brush4seed_config.site_tag_prefix)
            or tag in (ErrorTag.INVALID_TRACKER, )
        ]
        if useless_tags:
            torrent_actions.append(
                RemoveTagTorrentAction(torrent._torrent_hash, useless_tags)
            )
            torrent_logs.append(
                TorrentLog.new_log(
                    site_type or "-",
                    torrent._torrent_hash,
                    description=f"删除多余标签，{useless_tags}",
                    torrent_name=torrent.name,
                )
            )
        return torrent_actions, torrent_logs

    @staticmethod
    def _check_tracker(torrent):
        # 根据tracker状态检查做种是否有效，如果无效，进行删除
        if TorrentService().check_trackers_validity(torrent.trackers):
            return [], []
        return [DeleteTorrentAction(torrent._torrent_hash, delete_file=False)], [
            TorrentLog.new_log(
                "-",
                torrent._torrent_hash,
                description="删除，原因：tracker汇报无效",
                torrent_name=torrent.name,
            )
        ]

# class SeedingTorrentsChecker(TorrentsCheckerBase):

#     def check(self):
#         pass

class DownloadingTorrentsChecker(TorrentsCheckerBase):
    def check(self, brush4seed_config):
        # 1. 检查免费是否即将到期
        logger.info("check free end...")
        downloading_torrents = self.downloader_service.get_downloading_torrents()
        self.downloader_service.bulk_execute_actions(
            self._common_check(
                downloading_torrents,
                functools.partial(
                    self._check_free_end, brush4seed_config=brush4seed_config
                ),
            )
        )
        # 2. 检查是否下载缓慢
        logger.info("check slow download...")
        downloading_torrents = self.downloader_service.get_downloading_torrents()
        self.downloader_service.bulk_execute_actions(
            self._common_check(
                downloading_torrents,
                functools.partial(
                    self._check_slow_download, brush4seed_config=brush4seed_config
                ),
            )
        )

    @staticmethod
    def _check_free_end(torrent, brush4seed_config: Brush4SeedConfig = None):
        now = int(time.time())
        site_type = TorrentService.get_site_type(brush4seed_config, torrent)
        if site_type is None:
            return [], []
        site_config = SiteService.get_site_config(brush4seed_config.sites, site_type)
        if site_config is None or site_config.ensure_free is False:
            return [], []
        free_end_time = DataService.get_free_end_time(torrent._torrent_hash)
        # 非本插件添加的，会查不到免费截止时间，这里不进行处理
        if free_end_time is None:
            logger.warn(
                f"种子{torrent._torrent_hash}的免费结束时间未知，可能为手动添加的种子，无法确保免费结束前自动暂停该种子"
            )
            return [], []
        if free_end_time > (now - 3600):
            return [], []
        logger.info(f"种子{torrent._torrent_hash}免费即将到期，自动暂停下载")
        return [
            # TODO:可配置是删除还是暂停,空间不够时考虑优先删除这些种子
            StopTorrentAction(torrent._torrent_hash),
            TagTorrentAction(torrent._torrent_hash, ErrorTag.FREE_END),
        ], [
            TorrentLog.new_log(
                site_type,
                torrent._torrent_hash,
                description="暂停，原因：免费即将结束",
                torrent_name=torrent.name,
            )
        ]

    @staticmethod
    def _check_slow_download(torrent, brush4seed_config: Brush4SeedConfig = None):
        site_type = TorrentService.get_site_type(brush4seed_config, torrent)
        if site_type is None:
            logger.warn(f"种子{torrent._torrent_hash} 未知站点")
            return [], []
        site_config = SiteService.get_site_config(brush4seed_config.sites, site_type)
        if site_config is None or site_config.min_dl_speed == 0:
            return [], []
        if torrent.properties.dl_speed_avg > site_config.min_dl_speed:
            return [], []
        logger.info(f"种子{torrent._torrent_hash}平均下载速度小于设定值，自动暂停下载")
        return [
            # TODO:可配置是删除还是暂停,空间不够时考虑优先删除这些种子
            StopTorrentAction(torrent._torrent_hash),
            TagTorrentAction(torrent._torrent_hash, ErrorTag.SLOW_DL),
        ], [
            TorrentLog.new_log(
                site_type,
                torrent._torrent_hash,
                description="暂停，原因：下载速度低于设定值",
                torrent_name=torrent.name,
            )
        ]

class DeleteService:
    def __init__(self, brush4seed_config):
        self.brush4seed_config = brush4seed_config
        self.site_service = SiteService(self.brush4seed_config)
        self.downloader_service = DownloaderService(self.brush4seed_config)
    
    def _get_all_save_path(self)->Tuple[List[Path], Set[Path]]:
        all_save_path = []
        exclude_save_path = set()
        downloader_preferences = self.downloader_service.get_preferences()
        save_path = Path(downloader_preferences.save_path)
        all_save_path.append(save_path)
        exclude_save_path.add(save_path)
        exclude_path_key_list = ["export_dir", "export_dir_fin", "temp_path"]
        for exclude_path_key in exclude_path_key_list:
            exclude_path = getattr(downloader_preferences, exclude_path_key)
            if not exclude_path:
                continue
            exclude_save_path.add(Path(exclude_path))

        all_categories = self.downloader_service.get_all_categories()
        for _, cate_value in all_categories.items():
            cate_save_path = cate_value["savePath"]
            if cate_save_path:
                cate_save_path = Path(cate_save_path)
            else:
                cate_save_path = save_path / cate_value["name"]
            all_save_path.append(cate_save_path)
            exclude_save_path.add(cate_save_path)
        return all_save_path, exclude_save_path
    
    def _get_file2seed_list(self)->Dict[Path, List[TorrentDictionary]]:
        all_save_path, exclude_save_path = self._get_all_save_path()

        # 文件及其对应种子列表
        file2seed_list = defaultdict(list)
        all_torrents = self.downloader_service.get_all_torrents()
        for torrent in all_torrents:
            content_path = Path(torrent.content_path)
            if content_path.is_file():
                content_path = content_path.parent
            file2seed_list[content_path].append(torrent)
            exclude_save_path.add(content_path.parent)

        # 遍历所有文件
        for save_path in all_save_path:
            for sub_dir in save_path.glob("*"):
                if sub_dir in file2seed_list:
                    continue
                if sub_dir.is_file():
                    continue
                # is other save path or is exclude_save_path
                if sub_dir in exclude_save_path:
                    continue
                file2seed_list[sub_dir] = []
        return file2seed_list
    
    def _can_all_delete(self, torrent_list, site2left_seeding_size):
        for torrent in torrent_list:
            # 某个种子标记成了不能删除
            if TorrentService.is_label_no_delete(self.brush4seed_config, torrent):
                return False
            # 某个站点还没达到做种体积
            site_type = TorrentService.get_site_type(self.brush4seed_config, torrent)
            if site_type and site_type in site2left_seeding_size and site2left_seeding_size[site_type] > 0:
                return False
        return True

    def _exclude_no_delete(self, file2seed_list, site2seeding_size):
        # TODO: hr条件判断
        site2left_seeding_size = self.downloader_service.get_site2left_seeding_size(site2current_seeding_size=site2seeding_size)
        can_delete_file2seed_list = defaultdict(list)
        for file_path, seed_list in file2seed_list.items():
            #TODO: 路径白名单
            if not self._can_all_delete(seed_list, site2left_seeding_size):
                continue
            can_delete_file2seed_list[file_path] = seed_list
        return can_delete_file2seed_list
    
    def _get_content_size(self, file_path, torrent_list):
        # 不手动计算，直接取种子大小
        if torrent_list:
            # 找最大值，可能某个种子没有全下载，找下载最多的
            return max([torrent.size for torrent in torrent_list])
        # 手动计算
        content_size = 0
        for subfile in file_path.glob("**/*"):
            if subfile.is_dir():
                continue
            content_size += subfile.stat().st_size
        return content_size
    
    def _get_score_sum(self, torrent_list):
        score_sum = 0
        for torrent in torrent_list:
            site_type = TorrentService.get_site_type(self.brush4seed_config, torrent)
            if not site_type:
                continue
            score_sum += sort_key_for_seed(torrent, eval_sort_score=self.site_service.get_eval_sort_score_for_seed(site_type))
        return score_sum
    
    def _update_site2over_seeding_size(self, site2over_seeding_size:Dict, may_delete_file):
        new_site2over_seeding_size = {}
        for torrent in may_delete_file["seed_list"]:
            site_type = TorrentService.get_site_type(self.brush4seed_config, torrent)
            if not site_type or site_type not in site2over_seeding_size:
                continue
            new_over_seeding_size = site2over_seeding_size[site_type] - may_delete_file["size_in_byte"]
            if new_over_seeding_size <= 0:
                # rollback
                return False
            new_site2over_seeding_size[site_type] = new_over_seeding_size
        
        # commit
        for site_type, new_over_seeding_size in new_site2over_seeding_size.items():
            site2over_seeding_size[site_type] = new_over_seeding_size
        return True

    def _get_may_delete_file_list(self, file2seed_list, need_delete_size_in_byte, site2seeding_size):
        all_file_list = []
        for file_path, seed_list in file2seed_list.items():
            all_file_list.append({
                "path": file_path,
                "size_in_byte": self._get_content_size(file_path, seed_list),
                "seed_size": len(seed_list),
                "score_sum": self._get_score_sum(seed_list),
                "seed_list": seed_list,
            })
        all_file_list.sort(key=lambda f:f["score_sum"])
        may_delete_file_list = []
        may_delete_size_in_byte = 0
        site2over_seeding_size = self.downloader_service.get_site2over_seeding_size(site2current_seeding_size=site2seeding_size, only_over_seeding=False)
        for may_delete_file in all_file_list:
            can_delete = self._update_site2over_seeding_size(site2over_seeding_size, may_delete_file)
            # 删除这个种子后有站点达不到做种体积了
            if not can_delete:
                continue
            may_delete_file_list.append(may_delete_file)
            may_delete_size_in_byte += may_delete_file["size_in_byte"]
            # 达到预期删除大小
            if may_delete_size_in_byte >= need_delete_size_in_byte:
                break
        return may_delete_file_list
    
    def _dump_file_for_save_data(self, file_may_delete):
        return {
                "path": str(file_may_delete["path"]),
                "size_in_byte": str(file_may_delete["size_in_byte"]),
                "seed_size": file_may_delete["seed_size"],
                "score_sum": str(file_may_delete["score_sum"]),
                "seed_hash_list": [
                    seed._torrent_hash for seed in file_may_delete["seed_list"]
                ],
        }
    
    def delete(self):
        logger.info("【delete start】")
        logger.info(f"使用下载器:{self.brush4seed_config.downloader_name}")
        free_space_on_disk_in_bytes = self.downloader_service.get_free_space_on_disk() 
        need_delete_size_in_byte = self.brush4seed_config.get_left_disk_space_after_delete_in_bytes() - free_space_on_disk_in_bytes
        if need_delete_size_in_byte < 0:
            logger.info(f"剩余空间充足:{1.0*free_space_on_disk_in_bytes/Ti:.2f} TiB，无需删种")
            return
        logger.info(f"需要删种{1.0*need_delete_size_in_byte/Ti:.2f} TiB")
        #1. 找到各文件做种情况
        file2seed_list = self._get_file2seed_list()
        #2. 排除不能删除的
        site2seeding_size = self.downloader_service.get_site2seeding_size()
        file2seed_list = self._exclude_no_delete(file2seed_list, site2seeding_size)
        #3. 计算候选删除文件列表
        may_delete_file_list = self._get_may_delete_file_list(file2seed_list, need_delete_size_in_byte, site2seeding_size)
        DataService.save_may_delete_file_list([
            self._dump_file_for_save_data(may_delete_file) for may_delete_file in may_delete_file_list
        ])
        #4. 自动删除？
        if self.brush4seed_config.confirm_delete:
            logger.info("开始自动删除")
            torrent_actions = []
            for may_delete_file in may_delete_file_list:
                file_path = may_delete_file["path"]
                logger.info(f"自动删除 {file_path}")
                seed_list = may_delete_file["seed_list"]
                if seed_list:
                    # 通过删种触发删除文件
                    torrent_actions.append(DeleteTorrentAction(seed_list[0]._torrent_hash))
                else:
                    # 直接删除文件
                    shutil.rmtree(file_path, ignore_errors=True)
            self.downloader_service.bulk_execute_actions(torrent_actions)
        logger.info("【delete end】")

class BrushService:
    def __init__(self, brush4seed_config):
        self.brush4seed_config = brush4seed_config

        self.site_service = SiteService(self.brush4seed_config)
        self.downloader_service = DownloaderService(
            self.brush4seed_config,
        )

    def brush(self):
        logger.info("「start」")
        logger.info(f"使用下载器:{self.brush4seed_config.downloader_name}")
        # check site tag
        logger.info("「check all managed torrents」")
        AllTorrentsChecker(self.downloader_service).check(self.brush4seed_config)
        # # 1. check seeding torrents
        # logger.info("「check seeding torrents」")
        # SeedingTorrentsChecker(self.downloader_service).check()
        # 2. check downloading torrents
        logger.info("「check 「downloading torrents」")
        DownloadingTorrentsChecker(self.downloader_service).check(
            self.brush4seed_config
        )
        # 3. get torrent for brush
        logger.info("「get torrent for brush」")
        torrents_to_add = self._get_torrents_for_brush()
        # 3.5 delete some torrents when no enough disk spaces
        torrents_size_to_add = get_total_size(torrents_to_add)
        new_free_space_on_disk = (
            self.downloader_service.get_free_space_on_disk() - torrents_size_to_add
        )
        if (
            len(torrents_to_add) > 0
            and new_free_space_on_disk
            < self.brush4seed_config.get_left_disk_space_bytes()
        ):
            logger.info(
                "「there is no enough disk space, end」"
            )
            return
            #space_need_to_clean = (
            #    self.brush4seed_config.get_left_disk_space_bytes()
            #    - new_free_space_on_disk
            #)
            #self._clean_space(space_need_to_clean)
        # 4. add torrent to downloader
        if torrents_to_add:
            logger.info("「send torrent to downloader」")
            self._add_torrent_to_downloader(torrents_to_add)

        logger.info("「end」")

    def _get_torrents_for_brush(
        self,
    ) -> List[TorrentInfo]:
        downloading_torrents = self.downloader_service.get_downloading_torrents()
        # 过滤掉暂停下载的种子
        downloading_torrents = [torrent for torrent in downloading_torrents if torrent.state != "pausedDL"]
        downloading_cnt = len(downloading_torrents)
        
        cnt_for_add = self.brush4seed_config.max_downloads - downloading_cnt
        if cnt_for_add <= 0:
            logger.info(
                f"当前下载：{downloading_cnt} > 允许的最大同时下载数：{self.brush4seed_config.max_downloads}"
            )
            return []
        logger.info(
            f"当前下载数:{downloading_cnt} 允许的最大同时下载数:{self.brush4seed_config.max_downloads} 即将添加{cnt_for_add}个种子到下载器"
        )

        # 当前可以添加{cnt_for_add}个种子，将cnt_for_add分配给各个站点
        site2left_seeding_size = self.downloader_service.get_site2left_seeding_size()
        if site2left_seeding_size:
            # 有某些站点还没达到设置的保种体积，仅对这些站点进行刷流，按照相差的做种体积作为权重
            brush_sites = list(site2left_seeding_size.keys())
            site2choice_wieght = [
                site2left_seeding_size[site_type] for site_type in brush_sites
            ]
        else:
            # 所有站点相同权重，随机分配
            # TODO:可配置的权重
            brush_sites = self.brush4seed_config.get_all_brush_enabled_sites()
            site2choice_wieght = [1 for _ in brush_sites]

        # 获取每个站点最新的种子(懒加载)
        site2new_torrents = {
            #site_type: self.site_service.get_torrents_for_brush(site_type)
            site_type: None
            for site_type in brush_sites
        }
        # 先按照权重抽取站点
        torrents_to_add = []
        # 按照站点的抽取结果取站点的种子，每次都取这个站点的第一个种子（前面已经对种子按照sort.sort_key_for_brush进行排序了）
        for site_type in choice_with_weight(brush_sites,site2choice_wieght):
            # 所有站点的种子都遍历了一遍，仍然没达到预期个数
            if self.no_site_new_torrent_left(site2new_torrents):
                break
            # 达到预期要求
            if len(torrents_to_add) >= cnt_for_add:
                break
            if site2new_torrents[site_type] is None:
                site2new_torrents[site_type] = self.site_service.get_torrents_for_brush(site_type)
            site_new_torrents = site2new_torrents[site_type]

            # 该站点没有新种子了，将机会顺延给下一个站点
            if not site_new_torrents:
                continue
            # 找到第一个没有下载过的种子，加入返回结果；或者全都下载过了，将机会顺延给下一个站点
            while True:
                new_torrent = site_new_torrents.pop(0)
                if self.downloader_service.is_not_downloaded_before(new_torrent):
                    setattr(new_torrent, "brush4seed_site_type", site_type)
                    torrents_to_add.append(new_torrent)
                    break
                elif not site_new_torrents:
                    break
        return torrents_to_add
    
    def no_site_new_torrent_left(self, site2new_torrents):
        for new_torrents in site2new_torrents.values():
            # 还没获取这个站点的种子
            if new_torrents is None:
                return False
            # 这个站点还有种子没遍历
            if new_torrents:
                return False
        return True

    def _clean_space(self, space_need_to_clean):
        site2current_seeding_size = defaultdict(int)
        site2torrents = defaultdict(list)
        # TODO:去除不满足hr要求的
        # TODO:优先删除未下载完成的
        for torrent in self.downloader_service.get_seeding_torrents():
            assert isinstance(torrent, TorrentDictionary)
            site_type = TorrentService.get_site_type(self.brush4seed_config, torrent)
            if site_type is None:
                continue
            site2current_seeding_size[site_type] += torrent.size
            site2torrents[site_type].append(torrent)
        site2over_seeding_size = self.downloader_service.get_site2over_seeding_size(
            site2current_seeding_size=site2current_seeding_size
        )
        # 各个站点的种子进行内部排序,每个站点的排序规则可以自定义
        for site_type, site_torrents in site2torrents.items():
            # 分数越大越重要，升序排列，这样越靠前的就是越没用的
            site_torrents.sort(
                key=functools.partial(
                    sort_key_for_seed,
                    eval_sort_score=self.site_service.get_eval_sort_score_for_seed(
                        site_type
                    ),
                ),
            )
        # 权重
        deleting_sites = len(site2over_seeding_size.keys())
        weights = [site2over_seeding_size[site_type] for site_type in deleting_sites]
        space_deleting_size = 0
        deleting_torrents = []
        deleting_torrent_cnt = 0
        torrent_total_cnt = sum(
            [len(site_torrents) for site_torrents in site2torrents.values()]
        )
        for current_site_type in choice_with_weight(deleting_sites, weights=weights):
            # 满足需要腾出的空间 or 全都删除了
            if (
                space_deleting_size >= space_need_to_clean
                or deleting_torrent_cnt >= torrent_total_cnt
            ):
                break
            site_torrents = site2torrents[current_site_type]
            if not site_torrents:
                continue
            deleting_torrent = site_torrents.pop(0)
            deleting_torrents.append(deleting_torrent)
            space_deleting_size += deleting_torrent.size
            deleting_torrent_cnt += 1
        if (
            deleting_torrent_cnt >= torrent_total_cnt
            and space_deleting_size < space_need_to_clean
        ):
            raise NoEnoughDiskSpace(
                "空间不足，即使删除全部做种中的种子，也不足以腾出空间"
            )
        logger.info("即将删除以下种子")
        torrent_actions = []
        for torrent in deleting_torrents:
            logger.info(f"{torrent._torrent_hash} name:{torrent.name}")
            torrent_actions.append(
                DeleteTorrentAction(torrent_hash=torrent._torrent_hash)
            )
        self.downloader_service.bulk_execute_actions(torrent_actions)

    def _add_torrent_to_downloader(self, torrents_to_add: List[TorrentInfo]):
        torrent_actions = []
        torrent_logs = []
        for torrent in torrents_to_add:
            free_end_time = StringUtils.str_to_timestamp(torrent.freedate)
            brush4seed_site_type = getattr(torrent, "brush4seed_site_type", None)
            if not self.brush4seed_config:
                logger.error(f"无法获取种子的site_type {torrent.to_dict()}")
                continue
            download_result = self.downloader_service.download_torrent(torrent)
            if download_result["type"] == "failed":
                logger.error(f"种子下载失败 {torrent.to_dict()}")
                continue
            try:
                if download_result["type"] == "magnet":
                    magnet_uri = download_result["magnet_uri"]
                    torrent_infohash = get_infohash_by_magnet_uri(magnet_uri)
                    add_action = AddTorrentAction(
                        magnet_uri=magnet_uri,
                        category=self.brush4seed_config.managed_cate,
                        tags=[
                            f"{self.brush4seed_config.site_tag_prefix}{brush4seed_site_type}",
                        ],
                    )
                elif download_result["type"] == "file":
                    torrent_file_path = download_result["file_path"]
                    torrent_file_path = Path(torrent_file_path)
                    if not torrent_file_path.exists():
                        logger.error(f"种子文件{torrent_file_path}不存在")
                        continue
                    torrent_infohash = get_infohash_v1(torrent_file_path)
                    add_action = AddTorrentAction(
                        torrent_file_path=torrent_file_path,
                        category=self.brush4seed_config.managed_cate,
                        tags=[
                            f"{self.brush4seed_config.site_tag_prefix}{brush4seed_site_type}",
                        ],
                    )
                else:
                    raise NotImplementedError
            except AddTorrentActionTypeError as ata_type_error:
                logger.error(f"{ata_type_error}")
                continue
            if free_end_time:
                DataService.save_free_end_time(torrent_infohash, free_end_time)
            torrent_logs.append(
                TorrentLog.new_log(
                    brush4seed_site_type,
                    torrent_infohash,
                    torrent_name=torrent.title,
                    description="新增",
                )
            )
            torrent_actions.append(add_action)
        self.downloader_service.bulk_execute_actions(torrent_actions)
        DataService.save_torrent_log(torrent_logs)
        # TODO:将其中几个文件的优先级调高


####################################################plugin###################################################


class Brush4Seed(_PluginBase):
    plugin_name = "刷流保种"
    plugin_desc = "刷流保种自动化"
    plugin_order = 99
    plugin_version = "0.1.5"
    plugin_author = "2017fighting"
    author_url = "https://github.com/2017fighting/MoviePilot-Plugins"

    lock = threading.Lock()
    brush4seed_config: Brush4SeedConfig

    def __init__(self):
        super().__init__()

    def init_plugin(self, config: dict = None):
        self.brush4seed_config = Brush4SeedConfig.init_by_plugin_config(config)
        self.brush_service = None
        self.delete_service = None
        if self.brush4seed_config.downloader_name:
            self.brush_service = BrushService(self.brush4seed_config)
            self.delete_service = DeleteService(self.brush4seed_config)
        self.form_service = FormService(self.brush4seed_config)
        self.page_service = PageService(self.brush4seed_config)

    def get_state(self):
        return True

    @staticmethod
    def get_command() -> List[Dict[str, Any]]:
        return []

    def get_api(self) -> List[Dict[str, Any]]:
        def _clear_log():
            DataService.clear_torrent_logs()
            DataService.clear_may_delete_file_list()
            return schemas.Response(success=True, message="成功")

        def _debug_data():
            return schemas.Response(
                success=True,
                data={
                    "config": self.brush4seed_config,
                    "torrents": DataService.get_torrents(),
                    "torrent_los": DataService.get_torrent_logs(),
                    "snapshot": DataService.get_snapshot(),
                    "may_delete_file_list": DataService.get_may_delete_file_list(),
                },
            )

        return [
            {
                "path": "/clear_log",
                "endpoint": _clear_log,
                "methods": ["GET"],
                "summary": "清空torrent log",
                "description": "清空torrent log",
            },
            {
                "path": "/debug",
                "endpoint": _debug_data,
                "methods": ["GET"],
                "summary": "debug",
                "description": "debug",
            },
        ]

    def get_form(self) -> Tuple[List[dict], Dict[str, Any]]:
        components = self.form_service.get_components()
        model = self.form_service.get_model()
        return (components, model)

    def get_page(self) -> List[dict]:
        torrent_logs = DataService.get_torrent_logs()
        may_delete_file_list = DataService.get_may_delete_file_list()
        seeding_torrents = []
        if self.brush_service:
            seeding_torrents = self.brush_service.downloader_service.get_all_torrents()
        today_snapshot = DataService.get_snapshot()
        return self.page_service.get_page(
            torrent_logs, may_delete_file_list, seeding_torrents, today_snapshot
        )

    def _brush(self, fire_once=False):
        with self.lock:
            self.brush_service.brush()
        # 立即执行，执行完成后删除立即执行的job
        if fire_once:
            Scheduler().remove_plugin_job(
                self.__class__.__name__, job_id="brush4seed_fire_once"
            )
    
    def _delete(self, fire_once=False):
        with self.lock:
            self.delete_service.delete()
        if fire_once:
            Scheduler().remove_plugin_job(
                self.__class__.__name__, job_id="brush4seed_delete_fire_once"
            )

    def _get_brush_service(self)->List[Dict[str, Any]]:
        brush_services = []
        # 定时运行
        if self.brush4seed_config.plugin_enable:
            for cron_idx, cron_expr in enumerate(self.brush4seed_config.cron_expr_list):
                brush_services.append(
                    {
                        "id": f"brush4seed_interval_{cron_idx}",
                        "name": f"刷流保种-定时刷流_{cron_idx}",
                        "func": functools.partial(self._brush, fire_once=False),
                        "trigger": CronTrigger.from_crontab(
                            cron_expr,
                            timezone=pytz.timezone(settings.TZ),
                        ),
                    }
                )

        # 立即执行一次，执行完后删除这个job
        if self.brush4seed_config.run_now:
            brush_services.append(
                {
                    "id": "brush4seed_fire_once",
                    "name": "刷流保种-刷流一次",
                    "func": functools.partial(self._brush, fire_once=True),
                    "trigger": DateTrigger(
                        run_date=datetime.now(tz=pytz.timezone(settings.TZ))
                        + timedelta(seconds=3),
                        timezone=pytz.timezone(settings.TZ),
                    ),
                }
            )
            self.brush4seed_config.run_now = False
            self._update_config()
        return brush_services
    
    def _get_delete_service(self) -> List[Dict[str, Any]]:
        delete_services = []
        if self.brush4seed_config.delete_cron_enable:
            for cron_idx, cron_expr in enumerate(self.brush4seed_config.cron_expr_list):
                delete_services.append(
                    {
                        "id": f"brush4seed_delete_interval_{cron_idx}",
                        "name": f"刷流保种-定时删种_{cron_idx}",
                        "func": functools.partial(self._delete, fire_once=False),
                        "trigger": CronTrigger.from_crontab(
                            cron_expr,
                            timezone=pytz.timezone(settings.TZ),
                        ),
                    }
                )
        
        if self.brush4seed_config.delete_run_now:
            delete_services.append({
                "id": "brush4seed_delete_fire_once",
                "name": "刷流保种-删种一次",
                "func": functools.partial(self._delete, fire_once=True),
                "trigger": DateTrigger(
                    run_date=datetime.now(tz=pytz.timezone(settings.TZ))
                    + timedelta(seconds=3),
                    timezone=pytz.timezone(settings.TZ),
                ),
            })
            self.brush4seed_config.delete_run_now = False
            self._update_config()
        return delete_services
        

    def get_service(self) -> List[Dict[str, Any]]:
        """
        注册插件公共服务
        """
        plugin_services = []
        plugin_services.append(
            {
                "id": "brush4seed_daily_snapshot",
                "name": "刷流保种-每日快照",
                "func": functools.partial(
                    SnapShotService.snapshot, brush4seed_config=self.brush4seed_config
                ),
                "trigger": CronTrigger.from_crontab(
                    "0 0 * * *",
                    timezone=pytz.timezone(settings.TZ),
                ),
            }
        )
        # 刷流服务
        plugin_services.extend(self._get_brush_service())
        # 删种服务
        plugin_services.extend(self._get_delete_service())
        return plugin_services

    def _update_config(self):
        self.update_config(self.brush4seed_config.dump_to_plugin_config())

    def stop_service(self):
        Scheduler().remove_plugin_job(pid=self.__class__.__name__)
