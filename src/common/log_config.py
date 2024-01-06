import logging

logging.basicConfig(level=logging.INFO,
                        format = logging.Formatter('%(name)-20s - %(asctime)s - %(levelname)s - %(message)s'),
                        datefmt='%m-%d %H:%M')