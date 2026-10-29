from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import optuna
    from optuna.exceptions import TrialPruned
except Exception:
    optuna = None
    TrialPruned = RuntimeError


@dataclass
class PruningConfig:
    enabled: bool = False
    direction: str = "maximize"
    report_every: int = 1
    warmup_epochs: int = 0


class OptunaPruningCallback:
    def __init__(self, trial: Optional["optuna.trial.Trial"] = None, config: Optional[PruningConfig] = None):
        self.trial = trial
        self.config = config or PruningConfig()

    @property
    def active(self) -> bool:
        return self.trial is not None and self.config.enabled and optuna is not None

    def maybe_report_and_prune(self, metric_value: float, epoch: int) -> None:
        if not self.active:
            return
        if epoch < self.config.warmup_epochs:
            return
        if self.config.report_every > 1 and (epoch % self.config.report_every != 0):
            return
        self.trial.report(float(metric_value), step=int(epoch))
        if self.trial.should_prune():
            raise TrialPruned(f"Trial pruned at epoch={epoch}, metric={metric_value:.6f}")
