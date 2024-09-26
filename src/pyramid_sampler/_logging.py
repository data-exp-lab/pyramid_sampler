from __future__ import annotations

import logging

sampler_log = logging.getLogger("pyramid_sampler")
sampler_log.setLevel(logging.INFO)

_formatter = logging.Formatter("%(name)s : [%(levelname)s ] %(asctime)s:  %(message)s")

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(_formatter)
sampler_log.addHandler(stream_handler)
