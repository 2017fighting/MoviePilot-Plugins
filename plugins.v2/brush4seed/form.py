from typing import Any, Dict, List

from app.helper.downloader import DownloaderHelper
from app.helper.sites import SitesHelper
from app.plugins.brush4seed.config import brush4seed_config
from app.plugins.brush4seed.vuetify import (
    VBadge,
    VCol,
    VForm,
    VRow,
    VSelect,
    VSwitch,
    VTab,
    VTabs,
    VTextarea,
    VTextField,
    VWindow,
    VWindowItem,
)
from app.utils.string import StringUtils


class FormService:
    def __init__(self):
        pass

    def get_components(self) -> List[dict]:
        site_tabs, site_windows = self._get_site_tabs_and_windows()
        components = [
            VForm(
                VRow(
                    VCol(VSwitch(model="plugin_enable", label="启用插件")),
                    VCol(VSwitch(model="run_now", label="立即运行一次")),
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
                            label="确保剩余磁盘空间大于，单位TiB",
                            type="number",
                            placeholder="15",
                        ),
                    ),
                ),
                VRow(
                    VCol(
                        VTextField(
                            model="cron_expr_list",
                            label="执行周期",
                            placeholder="支持多个，逗号分隔，如果想工作日和假期不同频率，则类似于：15/45 * * * 1-5,15 * * * 0,6",
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
            site_config = brush4seed_config.sites.get(site_type)
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
        model = brush4seed_config.dump_to_plugin_config()
        # site_indexers = SitesHelper().get_indexers()
        # _site_tab = None
        # if site_indexers:
        #     _site_tab = StringUtils.get_url_domain(site_indexers[0].get("domain"))
        # model["_site_tab"] = _site_tab
        # model["_site_tab"] = ""
        return model
