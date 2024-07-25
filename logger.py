import logging
import time
import os
from multiprocessing import Lock

class Logger():
    def __init__(self, console_log=False, file_logging=False, file_URI=None, level=logging.DEBUG, override=False, log_name="baselog"):
        self.log_name = log_name
        self.console_log = console_log
        self.file_logging = file_logging
        self.lock = Lock()

        if file_logging:
            if file_URI is None:
                file_URI = "{}".format(self.log_name) + "_log_{}".format(time.strftime("%Y-%m-%d_%H-%M-%S")) + ".txt"
            else:
                if os.path.exists(file_URI) and not override:
                    raise NameError("Log File already exists! Try setting override flag")
                if os.path.exists(file_URI) and override:
                    os.remove(file_URI)
            file_URI = file_URI.replace(" ", "_").replace(":", "-")
            self.file_URI = file_URI
            logging.basicConfig(filename=file_URI, encoding='utf-8', level=level, format='%(asctime)s %(message)s')

    def warning(self, skk, printout=True):  # yellow
        with self.lock:
            if printout and self.console_log:
                print("\033[93m{} WARNING:\033[00m" .format(self.time_str()), "\033[93m {}\033[00m".format(skk))
            if self.file_logging:
                logging.warning(skk)

    def error(self, skk, printout=True):  # red
        with self.lock:
            if printout and self.console_log:
                print("\033[91m{} ERROR:\033[00m" .format(self.time_str()), "\033[91m {}\033[00m".format(skk))
            if self.file_logging:
                logging.error(skk)

    def fail(self, skk, printout=True):  # red
        with self.lock:
            if printout and self.console_log:
                print("\033[91m{} FATAL:\033[00m" .format(self.time_str()), "\033[91m {}\033[00m".format(skk))
            if self.file_logging:
                logging.exception(skk)

    def passing(self, skk, printout=True):  # green
        with self.lock:
            if printout and self.console_log:
                print("\033[92m{} \033[00m" .format(self.time_str()), "\033[92m {}\033[00m".format(skk))
            if self.file_logging:
                logging.info(skk)

    def passingblue(self, skk, printout=True):  # blue
        with self.lock:
            if printout and self.console_log:
                print("\033[96m{} \033[00m" .format(self.time_str()), "\033[96m {}\033[00m".format(skk))
            if self.file_logging:
                logging.info(skk)

    def info(self, skk, printout=True):  # blue
        with self.lock:
            if printout and self.console_log:
                print("\033[94m{} INFO:\033[00m" .format(self.time_str()), "\033[94m {}\033[00m".format(skk))
            if self.file_logging:
                logging.debug(skk)

    def time_str(self):
        return time.strftime("%d:%m:%Y:%H:%M:%S", time.localtime())
