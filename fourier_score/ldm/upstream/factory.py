"""The supported unconditional modules do not require dynamic factories."""


def instantiate_from_config(config):
    raise ValueError("Dynamic conditioning/first-stage factories are not supported")
