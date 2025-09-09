from typing import List, Optional, Tuple

import i6_core.rasr as rasr


def _has_image(config: rasr.RasrConfig, post_config: Optional[rasr.RasrConfig]):
    res = config._get("image") is not None
    res = res or (post_config is not None and post_config._get("image") is not None)
    return res


def find_arpa_lms(
    lm_config: rasr.RasrConfig, lm_post_config: Optional[rasr.RasrConfig] = None
) -> List[Tuple[rasr.RasrConfig, rasr.RasrConfig]]:
    result = []

    if lm_config.type == "ARPA":
        if not _has_image(lm_config, lm_post_config):
            result.append((lm_config, lm_post_config))
    elif lm_config.type == "combine":
        for i in range(1, lm_config.num_lms + 1):
            sub_lm_config = lm_config[f"lm-{i}"]
            sub_lm_post_config = lm_post_config[f"lm-{i}"] if lm_post_config is not None else None
            result += find_arpa_lms(sub_lm_config, sub_lm_post_config)

    return result
