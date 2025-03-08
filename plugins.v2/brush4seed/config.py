from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from app.helper.sites import SitesHelper
from app.plugins.brush4seed.tracker import TrackerService
from app.utils.singleton import Singleton
from app.utils.string import StringUtils

# "15/45 * * * 1-5"
DEFAULT_CRON_EXPR = "10 * * * *"
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
    managed_tag: str = "brush4seed"  # 该插件所有操作仅会涉及这个tag下的种子
    site_tag_prefix: str = "site:"

    plugin_enable: bool = False
    run_now: bool = False
    downloader_name: str = None
    cron_expr_list: List[str] = None
    max_downloads: int = 6  # 同时下载任务数
    left_disk_space: float = 1  # 确保空闲空间大于这个值 单位TiB
    sites: Dict[str, Brush4SeedSiteConfig] = None  # 只有enable=true才加载进来

    def init_by_plugin_config(self, plugin_config: dict):
        self.plugin_enable = plugin_config.get("plugin_enable", False)
        self.run_now = plugin_config.get("run_now", False)
        self.downloader_name = plugin_config.get("downloader_name")
        cron_expr_list = plugin_config.get("cron_expr_list")
        if not cron_expr_list:
            cron_expr_list = [DEFAULT_CRON_EXPR]
        else:
            cron_expr_list = cron_expr_list.split("|")
        self.cron_expr_list = cron_expr_list
        self.max_downloads = int(plugin_config.get("max_downloads", 1))
        self.left_disk_space = float(plugin_config.get("left_disk_space", 0))

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
            elif k == "cron_expr_list":
                result[k] = "|".join(v)
            elif k == "left_disk_space":
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


brush4seed_config = Brush4SeedConfig()
