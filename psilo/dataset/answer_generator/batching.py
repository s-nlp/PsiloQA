from utils.constants import SEED


def assign_runners_by_language(
    samples: list[dict],
    choose_runner_for_lang,  # callable(lang, seed) -> runner or None
    seed: int | None = None,
) -> dict[str, list[tuple[int, dict]]]:
    """
    Returns mapping runner_id -> list of (index, sample).
    """
    buckets: dict[str, list[tuple[int, dict]]] = {}
    for i, s in enumerate(samples):
        lang = (s.get("language") or "").lower()
        r = choose_runner_for_lang(lang, seed=SEED + i)  # SEED + i for assigning random model for each sample
        if r is None:
            continue
        buckets.setdefault(r.runner_id, []).append((i, s))
    return buckets
