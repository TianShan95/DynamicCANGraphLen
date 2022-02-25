import logging


def mylogger():
    # LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    # logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    # 输出到 文件
    # handler = logging.FileHandler("log.txt")
    # handler.setLevel(logging.DEBUG)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # handler.setFormatter(formatter)

    # 输出到 屏幕
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)

    # 加入输出
    # logger.addHandler(handler)
    logger.addHandler(console)

    # logger.info("Start print log")
    # logger.debug("Do something")
    # logger.warning("Something maybe fail.")
    # logger.info("Finish")

    return logger

logger = mylogger()