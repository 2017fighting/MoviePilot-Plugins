from typing import Optional
from urllib.parse import urlparse

from app.chain.torrents import TorrentsChain
from app.db.site_oper import SiteOper
from app.log import logger
from app.plugins.brush4seed import sort
from app.plugins.brush4seed.config import Brush4SeedSiteConfig, brush4seed_config


class SiteService:
    def __init__(self):
        self.site_oper = SiteOper()
        self.torrents_chain = TorrentsChain()

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
        site_config = self.get_site_config(site_type)
        assert site_config is not None
        if site_config.ensure_free is True:
            torrents = [
                torrent for torrent in torrents if torrent.downloadvolumefactor == 0
            ]
        # TODO:其他筛选条件，例如hr、种子体积、种子名称
        # 排序
        for torrent in torrents:
            torrent.brush4seed_score = sort.sort_key_for_brush(torrent)
        logger.debug(f"站点:{site_type}\n{torrents=}")
        torrents.sort(key=lambda x: x.brush4seed_score, reverse=True)
        return torrents

    def get_eval_sort_score_for_seed(self, site_type: str):
        site_config = self.get_site_config(site_type)
        if not site_config or site_config.custom_sort_eval_for_seed is None:
            return sort.DEFAULT_EVAL_FOR_SEED
        return site_config.custom_sort_eval_for_seed

    @classmethod
    def get_site_config(cls, site_type) -> Optional[Brush4SeedSiteConfig]:
        return brush4seed_config.sites.get(site_type)

    @classmethod
    def get_site_type_by_tracker(cls, tracker_url: str) -> Optional[str]:
        netloc = urlparse(tracker_url).netloc
        for site_type, site_config in brush4seed_config.sites.items():
            if netloc in site_config.trackers:
                return site_type
        return None
