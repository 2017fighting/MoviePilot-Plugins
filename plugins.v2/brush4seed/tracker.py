from typing import List


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


class TrackerService:

    @classmethod
    def get_default_trackers(cls, site_type):
        for site_class in SiteConfig.__subclasses__():
            if site_class.site_type != site_type:
                continue
            return site_class.trackers
        return []
