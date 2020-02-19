import argparse
import torch


class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def merge(self,argv, config_dict=None ):
        if config_dict is None:
            # args = self.parser.parse_args(argv)
            args, unknown = self.parser.parse_known_args(argv)
            config_dict = args.__dict__
        for key in config_dict.keys():
            setattr(self, key, config_dict[key])

