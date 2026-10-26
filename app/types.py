"""Small data types shared across the app."""

from dataclasses import dataclass, field


@dataclass
class Completion:
    """What a model returned, plus token counts for cost tracking."""

    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    failures: list[str] = field(default_factory=list)  # calls that failed before this one worked


@dataclass
class TraceStep:
    """One step a message went through, shown in the UI's 'under the hood' panel."""

    name: str
    detail: str
    ms: float = 0.0
    status: str = "ok"  # ok | warn | blocked | error


@dataclass
class Reply:
    """The assistant's answer to one customer message."""

    text: str
    trace: list[TraceStep] = field(default_factory=list)
    cost_usd: float = 0.0

    @property
    def total_ms(self) -> float:
        return sum(step.ms for step in self.trace)
