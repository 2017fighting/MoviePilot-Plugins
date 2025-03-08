import functools
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import pytz
from app.core.config import settings
from app.plugins import _PluginBase
from app.plugins.brush4seed.brush import BrushService
from app.plugins.brush4seed.config import brush4seed_config
from app.plugins.brush4seed.data import DataService
from app.plugins.brush4seed.form import FormService
from app.plugins.brush4seed.page import PageService
from app.plugins.brush4seed.snapshot import SnapShotService
from app.scheduler import Scheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger


class Brush4Seed(_PluginBase):
    plugin_name = "刷流保种"
    plugin_desc = "刷流保种自动化"
    plugin_order = 99
    plugin_version = "0.1.0"

    lock = threading.Lock()

    def __init__(self):
        super().__init__()

    def init_plugin(self, config: dict = None):
        brush4seed_config.init_by_plugin_config(config)
        self.brush_service = None
        if brush4seed_config.plugin_enable:
            self.brush_service = BrushService()
        self.form_service = FormService()
        self.page_service = PageService()

    def get_state(self):
        return brush4seed_config.plugin_enable

    @staticmethod
    def get_command() -> List[Dict[str, Any]]:
        return []

    def get_api(self) -> List[Dict[str, Any]]:
        return []

    def get_form(self) -> Tuple[List[dict], Dict[str, Any]]:
        components = self.form_service.get_components()
        model = self.form_service.get_model()
        return (components, model)

    def get_page(self) -> List[dict]:
        torrent_logs = DataService.get_torrent_logs()
        seeding_torrents = []
        if self.brush_service:
            seeding_torrents = self.brush_service.downloader_service.get_all_torrents()
        today_datetime = datetime.now(tz=pytz.timezone(settings.TZ))
        today_snapshot = DataService.get_snapshot(today_datetime)
        return self.page_service.get_page(
            torrent_logs, seeding_torrents, today_snapshot
        )

    def _brush(self, fire_once=False):
        with self.lock:
            self.brush_service.brush()
        # 立即执行，执行完成后删除立即执行的job
        if fire_once:
            Scheduler().remove_plugin_job(
                self.__class__.__name__, job_id="brush4seed_fire_once"
            )

    def get_service(self) -> List[Dict[str, Any]]:
        """
        注册插件公共服务
        """
        if not brush4seed_config.plugin_enable:
            return []
        plugin_services = []
        # 立即执行一次，执行完后删除这个job
        if brush4seed_config.run_now:
            plugin_services.append(
                {
                    "id": "brush4seed_fire_once",
                    "name": "刷流保种-立即执行",
                    "func": functools.partial(self._brush, fire_once=True),
                    "trigger": DateTrigger(
                        run_date=datetime.now(tz=pytz.timezone(settings.TZ))
                        + timedelta(seconds=3),
                        timezone=pytz.timezone(settings.TZ),
                    ),
                }
            )
        for cron_idx, cron_expr in enumerate(brush4seed_config.cron_expr_list):
            plugin_services.append(
                {
                    "id": f"brush4seed_interval_{cron_idx}",
                    "name": f"刷流保种-定时执行_{cron_idx}",
                    "func": functools.partial(self._brush, fire_once=False),
                    "trigger": CronTrigger.from_crontab(
                        cron_expr,
                        timezone=pytz.timezone(settings.TZ),
                    ),
                }
            )
        plugin_services.append(
            {
                "id": "brush4seed_daily_snapshot",
                "name": "刷流保种-每日快照",
                "func": SnapShotService.snapshot,
                "trigger": CronTrigger.from_crontab(
                    "0 0 * * *",
                    timezone=pytz.timezone(settings.TZ),
                ),
            }
        )
        brush4seed_config.run_now = False
        self._update_config()
        return plugin_services

    def _update_config(self):
        self.update_config(brush4seed_config.dump_to_plugin_config())

    def stop_service(self):
        Scheduler().remove_plugin_job(pid=self.__class__.__name__)
