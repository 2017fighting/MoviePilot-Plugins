from app.plugins.brush4seed.config import brush4seed_config
from app.plugins.brush4seed.data import DataService, TorrentSnapShot
from app.plugins.brush4seed.downloader import DownloaderService


class SnapShotService:
    @classmethod
    def snapshot(cls):
        snapshots = []
        downloader_service = DownloaderService(brush4seed_config.downloader_name)
        all_torrents = downloader_service.get_all_torrents()
        for torrent in all_torrents:
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
        DataService.save_snapshot(snapshots)
