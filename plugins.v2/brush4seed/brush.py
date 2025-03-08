import functools
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Tuple

from qbittorrentapi import TorrentDictionary

from app.core.context import TorrentInfo
from app.log import logger
from app.plugins.brush4seed.config import brush4seed_config
from app.plugins.brush4seed.data import DataService, TorrentLog
from app.plugins.brush4seed.db import PluginDao
from app.plugins.brush4seed.downloader import (
    AddTorrentAction,
    DeleteTorrentAction,
    DownloaderService,
    StopTorrentAction,
    TagTorrentAction,
    TorrentActionBase,
)
from app.plugins.brush4seed.error import AddTorrentActionTypeError, NoEnoughDiskSpace
from app.plugins.brush4seed.site import SiteService
from app.plugins.brush4seed.sort import sort_key_for_seed
from app.plugins.brush4seed.tag import ErrorTag
from app.plugins.brush4seed.torrent import (
    TorrentService,
    get_infohash_by_magnet_uri,
    get_infohash_v1,
    get_total_size,
)
from app.plugins.brush4seed.util import choice_with_weight
from app.utils.string import StringUtils


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
            actions, torrent_logs = check_func(torrent)
            if not actions:
                continue
            torrent_actions.extend(actions)
            torrent_logs.extend(torrent_logs)
        DataService.save_torrent_log(torrent_logs)
        return torrent_actions


class AllTorrentsChecker(TorrentsCheckerBase):

    def check(self):
        # 是否有site:xxx标签，没有的补全
        logger.info("check site tag...")
        seeding_torrents = self.downloader_service.get_all_torrents()
        self.downloader_service.bulk_execute_actions(
            self._common_check(seeding_torrents, self._check_site_tag)
        )

    @staticmethod
    def _check_site_tag(torrent):
        site_tag = TorrentService.get_site_tag(torrent)
        if site_tag is not None:
            return [], []
        first_tracker = TorrentService.get_first_tracker(torrent)
        site_type = SiteService.get_site_type_by_tracker(first_tracker)
        if site_type:
            new_tag = f"{brush4seed_config.site_tag_prefix}{site_type}"
        else:
            new_tag = ErrorTag.INVALID_TRACKER
        return [TagTorrentAction(torrent._torrent_hash, new_tag)], [
            TorrentLog.new_log(
                site_type,
                torrent._torrent_hash,
                description=f"手动添加，新增标签{new_tag}",
                torrent_name=torrent.name,
            )
        ]


class SeedingTorrentsChecker(TorrentsCheckerBase):

    def check(self):
        # 1. tracker是否有效
        logger.info("check tracker...")
        seeding_torrents = self.downloader_service.get_seeding_torrents()
        self.downloader_service.bulk_execute_actions(
            self._common_check(seeding_torrents, self._check_tracker)
        )

    @staticmethod
    def _check_tracker(torrent):
        # 根据tracker状态检查做种是否有效，如果无效，进行删除
        if TorrentService().check_trackers_validity(torrent.trackers):
            return [], []
        return [DeleteTorrentAction(torrent._torrent_hash)], [
            TorrentLog.new_log(
                "-",
                torrent._torrent_hash,
                description="删除，原因：tracker汇报无效",
                torrent_name=torrent.name,
            )
        ]


class DownloadingTorrentsChecker(TorrentsCheckerBase):
    def check(self):
        # 1. 检查免费是否即将到期
        logger.info("check free end...")
        downloading_torrents = self.downloader_service.get_downloading_torrents()
        self.downloader_service.bulk_execute_actions(
            self._common_check(downloading_torrents, self._check_free_end)
        )
        # 2. 检查是否下载缓慢
        logger.info("check slow download...")
        downloading_torrents = self.downloader_service.get_downloading_torrents()
        self.downloader_service.bulk_execute_actions(
            self._common_check(downloading_torrents, self._check_slow_download)
        )

    @staticmethod
    def _check_free_end(torrent):
        now = int(time.time())
        site_type = TorrentService.get_site_type(torrent)
        if site_type is None:
            return [], []
        site_config = SiteService.get_site_config(site_type)
        if site_config is None or site_config.ensure_free is False:
            return [], []
        free_end_time = PluginDao.get_free_end_time(torrent._torrent_hash)
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
    def _check_slow_download(torrent):
        site_type = TorrentService.get_site_type(torrent)
        if site_type is None:
            logger.warn(f"种子{torrent._torrent_hash} 未知站点")
            return [], []
        site_config = SiteService.get_site_config(site_type)
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


class BrushService:
    def __init__(self):
        self.site_service = SiteService()
        self.downloader_service = DownloaderService(
            downloader_name=brush4seed_config.downloader_name
        )

    def brush(self):
        logger.info("「start」")
        logger.info(f"使用下载器:{brush4seed_config.downloader_name}")
        # check site tag
        logger.info("「check all managed torrents」")
        AllTorrentsChecker(self.downloader_service).check()
        # 1. check seeding torrents
        logger.info("「check seeding torrents」")
        SeedingTorrentsChecker(self.downloader_service).check()
        # 2. check downloading torrents
        logger.info("「check 「downloading torrents」")
        DownloadingTorrentsChecker(self.downloader_service).check()
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
            and new_free_space_on_disk < brush4seed_config.get_left_disk_space_bytes()
        ):
            logger.info(
                "「there is no enough disk space, try to delete some torrents」"
            )
            space_need_to_clean = (
                brush4seed_config.get_left_disk_space_bytes() - new_free_space_on_disk
            )
            self._clean_space(space_need_to_clean)
        # 4. add torrent to downloader
        if torrents_to_add:
            logger.info("「send torrent to downloader」")
            self._add_torrent_to_downloader(torrents_to_add)

        logger.info("「end」")

    def _get_torrents_for_brush(
        self,
    ) -> List[TorrentInfo]:
        downloading_cnt = len(
            self.downloader_service.get_downloading_torrents(only_managed=False)
        )
        cnt_for_add = brush4seed_config.max_downloads - downloading_cnt
        if cnt_for_add <= 0:
            logger.info(
                f"当前下载：{downloading_cnt} > 允许的最大同时下载数：{brush4seed_config.max_downloads}"
            )
            return []
        logger.info(
            f"当前下载数:{downloading_cnt} 允许的最大同时下载数:{brush4seed_config.max_downloads} 即将添加{cnt_for_add}个种子到下载器"
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
            brush_sites = brush4seed_config.get_all_brush_enabled_sites()
            site2choice_wieght = None

        # 获取每个站点最新的种子
        site2new_torrents = {
            site_type: self.site_service.get_torrents_for_brush(site_type)
            for site_type in brush_sites
        }
        # 总共获取了多少个种子
        torrents_total_len = sum(
            len(torrents) for torrents in site2new_torrents.values()
        )
        # 先按照权重抽取站点
        brush_site_sequence = random.choices(
            brush_sites, weights=site2choice_wieght, k=torrents_total_len
        )
        torrents_to_add = []
        # 按照站点的抽取结果取站点的种子，每次都取这个站点的第一个种子（前面已经对种子按照sort.sort_key_for_brush进行排序了）
        for site_type in brush_site_sequence:
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
            # 达到预期要求
            if len(torrents_to_add) >= cnt_for_add:
                break
        return torrents_to_add

    def _clean_space(self, space_need_to_clean):
        site2current_seeding_size = defaultdict(int)
        site2torrents = defaultdict(list)
        # TODO:去除不满足hr要求的
        # TODO:优先删除未下载完成的
        for torrent in self.downloader_service.get_seeding_torrents():
            assert isinstance(torrent, TorrentDictionary)
            site_type = TorrentService.get_site_type(torrent)
            assert site_type is not None
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
            if not brush4seed_config:
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
                        tags=[
                            brush4seed_config.managed_tag,
                            f"{brush4seed_config.site_tag_prefix}{brush4seed_site_type}",
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
                        tags=[
                            brush4seed_config.managed_tag,
                            f"{brush4seed_config.site_tag_prefix}{brush4seed_site_type}",
                        ],
                    )
                else:
                    raise NotImplementedError
            except AddTorrentActionTypeError as ata_type_error:
                logger.error(f"{ata_type_error}")
                continue
            if free_end_time:
                PluginDao.save_free_end_time(torrent_infohash, free_end_time)
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
