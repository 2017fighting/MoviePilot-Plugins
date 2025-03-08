from collections import defaultdict
from dataclasses import asdict
from typing import List

from qbittorrentapi import TorrentInfoList

from app.db.plugindata_oper import PluginDataOper
from app.plugins.brush4seed.config import Gi, Ti
from app.plugins.brush4seed.data import TorrentLog, TorrentSnapShot
from app.plugins.brush4seed.site import SiteService
from app.plugins.brush4seed.torrent import TorrentService
from app.plugins.brush4seed.vuetify import VApexChart, VCol, VDataTableVirtual, VRow


class PageService:
    def __init__(self):
        self.plugin_data_oper = PluginDataOper()

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
            first_tracker = TorrentService.get_first_tracker(torrent)
            site_type = SiteService.get_site_type_by_tracker(first_tracker)
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
            first_tracker = TorrentService.get_first_tracker(torrent)
            site_type = SiteService.get_site_type_by_tracker(first_tracker)
            if not site_type:
                continue
            if torrent._torrent_hash not in infohash2snapshot:
                continue
            torrent_snapshot = infohash2snapshot[torrent._torrent_hash]
            assert isinstance(torrent_snapshot, TorrentSnapShot)
            today_uploads = torrent.uploaded - int(torrent_snapshot.uploaded)
            if today_uploads <= 0:
                continue
            site2today_uploads[site_type] += today_uploads
        return {
            site: 1.0 * today_uploads / Gi
            for site, today_uploads in site2today_uploads.items()
        }

    def get_page(
        self,
        torrent_logs: List[TorrentLog],
        all_torrents: TorrentInfoList,
        today_snapshot,
    ):
        site_type2seeding_TiB = self._get_site_type2seeding_size(all_torrents)
        seeding_sites = list(site_type2seeding_TiB.keys())
        seeding_sizes = [
            site_type2seeding_TiB[site_type] for site_type in seeding_sites
        ]

        site2today_uploads = self._get_site2today_uploads(today_snapshot, all_torrents)
        upload_sites = list(site2today_uploads.keys())
        uploads = [site2today_uploads[site_type] for site_type in upload_sites]

        page = [
            VRow(
                VCol(
                    self._get_pie_chart(
                        "做种体积(TiB)",
                        seeding_sites,
                        seeding_sizes,
                    )
                ),
                VCol(
                    self._get_pie_chart(
                        "今日上传(GiB)",
                        upload_sites,
                        uploads,
                    )
                ),
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
                    )
                )
            ),
        ]
        return [p.as_dict() for p in page]
