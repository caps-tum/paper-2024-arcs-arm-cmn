from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DSU:
    x: int
    y: int
    p: int
    coreids: Optional[Tuple[int,int]] = None

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.p == other.p

    def __hash__(self):
        return hash(f"{self.x}{self.y}{self.p}")


@dataclass
class Run:
    name: str
    df: pd.DataFrame
    binary: str = ""
    meta: str = ""

    def concat(self, other_run):
        self.df = pd.concat([self.df, other_run.df])
        return self

    def merge(self, other_run):
        _vals = np.array([self.df.values, other_run.df.values])
        self.df = pd.DataFrame(np.mean(_vals, axis=0))
        return self

    def set_name(self, name: str):
        self.name = name
        return self
