"""Pack ranked units into a token budget."""

from dataclasses import dataclass
from typing import Callable

from ragsplit.units import Unit

SEP = "\n\n"


@dataclass
class Packed:
    unit_ids: list[str]
    text: str
    truncated: bool  # the last unit was cut to fit the budget


def pack(
    ranked: list[Unit],
    budget: int,
    count: Callable[[str], int],
    truncate: Callable[[str, int], str],
) -> Packed:
    """Fill the context in rank order until `budget` tokens are used.

    A unit that overlaps text already packed (e.g. a piece and its parent
    paragraph) is skipped, so the higher-ranked one wins and no text is sent
    twice. The first unit that does not fit is truncated to the remaining room.
    """
    sep_cost = count(SEP)
    parts: list[str] = []
    ids: list[str] = []
    covered: set = set()
    used = 0
    for u in ranked:
        if u.covers() & covered:
            continue
        sep = sep_cost if parts else 0
        cost = count(u.text) + sep
        if used + cost <= budget:
            parts.append(u.text)
            ids.append(u.id)
            covered |= u.covers()
            used += cost
            continue
        room = budget - used - sep
        if room > 0:
            parts.append(truncate(u.text, room))
            ids.append(u.id)
            return Packed(ids, SEP.join(parts), True)
        break
    return Packed(ids, SEP.join(parts), False)
