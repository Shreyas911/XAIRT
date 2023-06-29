""" Low-level operations with Keras """
from __future__ import annotations

import warnings
warnings.simplefilter("ignore")

import tensorflow.keras as keras

# Get useful types from types.py
from XAIRT.backend.types import kModel

__all__ =["getLayerIndexByName"]

def getLayerIndexByName(model: kModel, layername: str) -> int:
	for idx, layer in enumerate(model.layers):
		if layer.name == layername: 
			return idx
	raise ValueError(f"layername: {layername} not found.")

	
