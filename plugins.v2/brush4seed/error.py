class Brush4SeedBaseException(Exception):
    pass


class AddTorrentActionTypeError(Brush4SeedBaseException):
    pass


class NoEnoughDiskSpace(Brush4SeedBaseException):
    pass


class ExternalError(Brush4SeedBaseException):
    # 本插件依赖的外部服务出错
    pass


class InvalidDownloader(Brush4SeedBaseException):
    # 下载器出错
    pass
