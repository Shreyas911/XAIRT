from __future__ import annotations

from XAIRT.backend.graph import getLayerIndexByName, useGradientTape, tf_to_numpy
from XAIRT.model.Trainer import TrainLR, TrainFullyConnectedNN
from XAIRT.model.XAI import XLR, XAIR
from XAIRT.utils.stats import correlation

__version__ = "1.0.0"