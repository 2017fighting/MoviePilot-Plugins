import math

from qbittorrentapi import TorrentDictionary

from app.core.context import TorrentInfo
from app.log import logger

# 为做种中的种子进行排序，考虑种子大小和做种人数(种子越大，做种人数越少，得分越多)
DEFAULT_EVAL_FOR_SEED = (
    """(t.size/ 1024.0/ 1024/ 1024)* (1 + math.sqrt(2) * math.pow(10, -1.0 * (p.seeds_total - 1) / 6))""",
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
        if torrent.seeders <= 0:
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
