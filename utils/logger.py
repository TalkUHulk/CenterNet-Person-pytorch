import logging
import os
import datetime
import time
__all__ = ["Logger"]


class Logger(object):
    def __init__(self, logdir, level=logging.DEBUG, file_prefix="Trainer"):

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        logging.basicConfig(level=level,  # 控制台打印的日志级别
                            filename=os.path.join(logdir, '{}_{}.log'.format(
                                file_prefix,
                                datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
                            )),
                            filemode='w',
                            format=
                            '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                            )
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        self.start = time.time()
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.secs = time.time() - self.start
        self.logger.info("===============>>>>>> Cost time:{} secs".format(self.secs))

    def log(self, msg, level="debug"):
        getattr(self.logger, level)(msg)
