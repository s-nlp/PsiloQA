import random
from typing import Dict, List, Optional, Tuple

from utils.constants import SEED


def assign_runners_by_language(
    samples: List[Dict],
    choose_runner_for_lang,  # callable(lang, seed) -> runner or None
    seed: Optional[int] = None,
) -> Dict[str, List[Tuple[int, Dict]]]:
    """
    Returns mapping runner_id -> list of (index, sample).
    """
    rnd = random.Random(seed)
    buckets: Dict[str, List[Tuple[int, Dict]]] = {}
    for i, s in enumerate(samples):
        lang = (s.get("language") or "").lower()
        r = choose_runner_for_lang(
            lang, seed=SEED + i
        )  # SEED + i for assigning random model for each sample
        if r is None:
            continue
        buckets.setdefault(r.runner_id, []).append((i, s))
    return buckets
