from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class MetricDistribution:
    min: float = 0.0
    max: float = 0.0
    mean: float = 0.0

    @classmethod
    def from_values(cls, values: List[float]) -> "MetricDistribution":
        if not values:
            return cls()
        return cls(min=min(values), max=max(values), mean=sum(values) / len(values))

    def to_dict(self) -> Dict[str, float]:
        return {"min": self.min, "max": self.max, "mean": self.mean}


@dataclass
class MemoryStats:
    st_count: int
    ltm_count: int
    hot_count: int
    cold_count: int
    pointer_count: int
    forget_count: int
    g_distribution: MetricDistribution
    c_distribution: MetricDistribution
    t_distribution: MetricDistribution
    source: str = "Memoryos"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "st_count": self.st_count,
            "ltm_count": self.ltm_count,
            "hot_count": self.hot_count,
            "cold_count": self.cold_count,
            "pointer_count": self.pointer_count,
            "forget_count": self.forget_count,
            "G_distribution": self.g_distribution.to_dict(),
            "C_distribution": self.c_distribution.to_dict(),
            "T_distribution": self.t_distribution.to_dict(),
            "source": self.source,
        }
