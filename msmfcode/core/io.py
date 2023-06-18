import yaml

from os.path import join

from msmfcode.core.logging import log


def load_yaml(path_yaml, file_name_yaml):
    with open(join(path_yaml, file_name_yaml)) as file:
        try:
            data = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            log.exception(exc)

    return data
