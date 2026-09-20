"""Who the transaction belongs to: a synthetic card, merchant and country (Step 5.2).

The Kaggle columns are the output of a PCA the publishers ran to anonymise the
data. No card number, merchant or country survived it, and velocity features
need something to count per. So the producer invents identities and attaches
them to each transaction.

Three properties make them useful rather than decorative:

**Stable.** A transaction's identity comes from hashing its id, not from a
running random number generator. The producer can be stopped and resumed with
--start, and Kafka can deliver a message twice; the same transaction must not
arrive as a different card the second time.

**Heavy-tailed.** Cards are drawn with weight 1/rank**0.5, so a few are used far
more than the rest, as real card traffic is. Spread every transaction evenly
over 5,000 cards and no card would ever be busy enough for a velocity feature
to notice. Over the 56,746-transaction test set the busiest card takes about
405 of them (0.7%), the top 1% of cards about 9%, and the median card 8.

**Independent of the fraud label.** This module never looks at `Class`. It
would be easy to put fraud on a handful of cards and have velocity "catch" it,
but that would only be measuring the generator. Because identities are
independent of the label, Phase 5 can show what velocity costs (extra reviews,
extra latency) and how it behaves, and must not claim it catches more fraud.
Real velocity features earn their place on real identities.
"""

from __future__ import annotations

import hashlib
from bisect import bisect
from functools import lru_cache
from itertools import accumulate

# Cards in the synthetic portfolio, and how sharply traffic concentrates on the
# busiest of them. See the module docstring for what these produce.
CARD_COUNT = 5_000
CARD_WEIGHT_EXPONENT = 0.5

# Where a card is usually used, and how often it is used elsewhere. A card
# suddenly transacting in two countries is a classic velocity signal, so the
# share is small enough that seeing it is unusual.
HOME_COUNTRIES = ("US", "GB", "DE", "FR", "CA", "AU", "IN", "BR")
FOREIGN_COUNTRIES = ("NG", "RU", "CN", "MX", "RO", "TR")
FOREIGN_SHARE = 0.03

MERCHANTS = (
    "corner-grocer",
    "fuel-stop",
    "coffee-bar",
    "online-marketplace",
    "streaming-service",
    "pharmacy",
    "airline",
    "hotel-group",
    "electronics-store",
    "gaming-credits",
    "ride-hailing",
    "gift-cards",
)


def digest(value: str, salt: bytes) -> int:
    """A stable 64-bit number for this string. The salt keeps the draws independent.

    Python's hash() is salted per process, so it would give the same transaction
    a different card on every run.
    """
    raw = hashlib.blake2b(value.encode("utf-8"), digest_size=8, person=salt).digest()
    return int.from_bytes(raw, "big")


def fraction(value: str, salt: bytes) -> float:
    """The same number as `digest`, mapped into [0, 1) for choosing from a distribution."""
    return digest(value, salt) / 2**64


@lru_cache(maxsize=1)
def card_weights() -> list[float]:
    """Cumulative shares of traffic per card, for picking one by rank."""
    weights = [1 / rank**CARD_WEIGHT_EXPONENT for rank in range(1, CARD_COUNT + 1)]
    total = sum(weights)
    return list(accumulate(weight / total for weight in weights))


def card_for(transaction_id: str) -> str:
    """The card that made this transaction. Busy cards are picked far more often."""
    rank = bisect(card_weights(), fraction(transaction_id, b"card"))
    return f"card-{min(rank, CARD_COUNT - 1):05d}"


def merchant_for(transaction_id: str) -> str:
    """Where it was spent. Uniform: merchants are not what velocity keys off here."""
    return MERCHANTS[digest(transaction_id, b"merchant") % len(MERCHANTS)]


def country_for(transaction_id: str, card_id: str) -> str:
    """The card's home country, or, for a few transactions, somewhere it is not."""
    if fraction(transaction_id, b"abroad") < FOREIGN_SHARE:
        return FOREIGN_COUNTRIES[digest(transaction_id, b"country") % len(FOREIGN_COUNTRIES)]
    return HOME_COUNTRIES[digest(card_id, b"home") % len(HOME_COUNTRIES)]


def entities_for(transaction_id: str) -> dict[str, str]:
    """The card, merchant and country for this transaction id, as Transaction fields."""
    card_id = card_for(transaction_id)
    return {
        "card_id": card_id,
        "merchant": merchant_for(transaction_id),
        "country": country_for(transaction_id, card_id),
    }
