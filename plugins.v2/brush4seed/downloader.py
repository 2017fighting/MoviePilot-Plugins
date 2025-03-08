import itertools
from collections import defaultdict
from os import PathLike
from pathlib import Path
from types import NoneType
from typing import Dict, List, Optional, Union

from qbittorrentapi import (
    ApplicationPreferencesDictionary,
    TorrentDictionary,
    TorrentInfoList,
)

from app.chain.download import DownloadChain
from app.core.cache import cached
from app.core.context import TorrentInfo
from app.helper.downloader import DownloaderHelper
from app.log import logger
from app.modules.qbittorrent.qbittorrent import Qbittorrent
from app.plugins.brush4seed.config import brush4seed_config
from app.plugins.brush4seed.error import AddTorrentActionTypeError, InvalidDownloader
from app.plugins.brush4seed.torrent import (
    TorrentService,
    get_infohash_by_magnet_uri,
    get_infohash_v1,
)
from app.plugins.brush4seed.util import chunks


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


class DeleteTorrentAction(EditTorrentActionBase):
    pass


class StopTorrentAction(EditTorrentActionBase):
    torrent_hash: str
    pass


class DownloaderService:
    downloader: Qbittorrent

    def __init__(self, downloader_name: str):
        self.download_chain = DownloadChain()
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
        return self.downloader.qbc.torrents_info(tag=brush4seed_config.managed_tag)

    def get_seeding_torrents(self) -> TorrentInfoList:
        return self.downloader.qbc.torrents_info(
            status_filter="seeding", tag=brush4seed_config.managed_tag
        )

    def get_site2seeding_size(self) -> Dict[str, int]:
        site2seeding_size = defaultdict(int)
        for torrent in self.get_seeding_torrents():
            assert isinstance(torrent, TorrentDictionary)
            site_type = TorrentService.get_site_type(torrent)
            assert site_type is not None
            site2seeding_size[site_type] += torrent.size
        return site2seeding_size

    def get_site2left_seeding_size(
        self, site2current_seeding_size=None
    ) -> Dict[str, int]:
        # 站点->该站点还差多少才能达到设置的保种体积(单位bytes)，仅返回相差值为正数的
        if site2current_seeding_size is None:
            site2current_seeding_size = self.get_site2seeding_size()
        site2target_seeding_size = brush4seed_config.get_site2target_seeding_size(
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
        self, site2current_seeding_size=None
    ) -> Dict[str, int]:
        # 站点->该站点比设置的保种体积多了多少(单位bytes)，仅返回多做种的站点
        if site2current_seeding_size is None:
            site2current_seeding_size = self.get_site2seeding_size()
        site2target_seeding_size = brush4seed_config.get_site2target_seeding_size(
            only_brush_enabled=False
        )
        site2over_seeding_size = {}
        for site_type, target_seeding_size in site2target_seeding_size.items():
            current_seeding_size = site2current_seeding_size.get(site_type) or 0
            over_seeding_size = current_seeding_size - target_seeding_size
            if over_seeding_size <= 0:
                continue
            site2over_seeding_size[site_type] = over_seeding_size
        return site2over_seeding_size

    def get_downloading_torrents(self, only_managed=True) -> TorrentInfoList:
        return self.downloader.qbc.torrents_info(
            status_filter="downloading",
            tag=brush4seed_config.managed_tag if only_managed else None,
        )

    def bulk_execute_actions(self, torrent_actions: List[TorrentActionBase]):
        tag_actions = []
        delete_actions = []
        stop_actions = []
        add_actions = []
        hashes2delete = set()
        for action in torrent_actions:
            if isinstance(action, DeleteTorrentAction):
                hashes2delete.add(action.torrent_hash)
                delete_actions.append(action)
            elif isinstance(action, TagTorrentAction):
                if action.torrent_hash in hashes2delete:
                    continue
                tag_actions.append(action)
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
        for delete_action_chunk in chunks(delete_actions, chunk_size):
            self._torrents_delete(delete_action_chunk)
        for tag_action_chunk in chunks(tag_actions, chunk_size):
            self._torrents_add_tags(tag_action_chunk)
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

    def _torrents_delete(self, torrent_delete_actions: List[DeleteTorrentAction]):
        self.downloader.qbc.torrents_delete(
            delete_files=True,
            torrent_hashes=[action.torrent_hash for action in torrent_delete_actions],
        )

    def _torrents_stop(self, torrent_stop_actions: List[StopTorrentAction]):
        self.downloader.qbc.torrents_stop(
            [action.torrent_hash for action in torrent_stop_actions]
        )

    # def get_server_tags(self):
    #     return self.downloader.qbc.torrent_tags.tags
