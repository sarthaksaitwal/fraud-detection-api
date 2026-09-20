"""Velocity rules: when recent activity overrides the model's decision (Step 5.5).

The model scores one transaction in isolation. It was trained on `V1`-`V28` and
`Amount` and knows nothing about the card, so a card making its tenth payment in
a minute looks exactly like a card making its first. That pattern is one of the
oldest fraud signals there is, and it is what these rules add.

Rules run after the model, on the features measured in src/features/velocity.py:

    model says approve + a rule fires  ->  review
    model says review or block         ->  unchanged

**Escalation only, and only as far as review.** A wrong block costs a real
customer their payment; the cost assumptions behind our thresholds price that at
$50 against $5 for an analyst's time (see notebooks/008). Velocity is a
suspicion, not proof, so it can ask for a human, never refuse a payment. Nor can
it ever soften a decision the model already made.

**The thresholds are what a plausible card does, not what fits this dataset.**
Our card ids are synthetic and independent of the fraud label (see
src/features/entities.py), so tuning thresholds against the labels here would
only be fitting noise. They describe behaviour a real cardholder rarely shows:
several payments in a minute, two countries in an hour, two payments seconds
apart. What the project can then honestly report is the cost -- how much extra
review this asks for -- not a lift in fraud caught.

Every rule that fires is named in the response and stored with the decision, so
an analyst opening a review sees why it was sent to them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from src.api.schemas import Decision, VelocityFeatures
from src.config import settings


@dataclass(frozen=True)
class Rule:
    """One reason to ask a human to look, given what the card has been doing."""

    name: str
    description: str
    fires: Callable[[VelocityFeatures], bool]


def velocity_rules() -> tuple[Rule, ...]:
    """The rules, with their thresholds read from settings each time.

    Built on call rather than at import, so changing a threshold in .env (or a
    test) takes effect without reimporting the module.
    """
    return (
        Rule(
            "many_in_a_minute",
            f"more than {settings.velocity_max_per_minute} transactions on this card in a minute",
            lambda v: v.count_1m > settings.velocity_max_per_minute,
        ),
        Rule(
            "many_in_an_hour",
            f"more than {settings.velocity_max_per_hour} transactions on this card in an hour",
            lambda v: v.count_1h > settings.velocity_max_per_hour,
        ),
        Rule(
            "several_countries",
            f"used in {settings.velocity_max_countries + 1} or more countries within an hour",
            lambda v: v.countries_1h > settings.velocity_max_countries,
        ),
        Rule(
            "back_to_back",
            f"another transaction on this card less than "
            f"{settings.velocity_min_gap_seconds:g}s earlier",
            lambda v: v.seconds_since_previous is not None
            and v.seconds_since_previous < settings.velocity_min_gap_seconds,
        ),
    )


def reasons_for(velocity: VelocityFeatures | None) -> list[str]:
    """The names of every rule that fires for these features, in rule order."""
    if velocity is None:
        return []
    return [rule.name for rule in velocity_rules() if rule.fires(velocity)]


def escalate(decision: Decision, reasons: Sequence[str]) -> Decision:
    """Review instead of approve when a rule fired. Never downgrades a decision."""
    return Decision.REVIEW if reasons and decision is Decision.APPROVE else decision


def describe(name: str) -> str:
    """What a rule name means, for logs and for whoever reads a review queue."""
    for rule in velocity_rules():
        if rule.name == name:
            return rule.description
    return name
