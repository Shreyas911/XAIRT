import numpy as np

import warnings
import sys
import os
import pathlib

import tensorflow as tf
import tensorflow.keras as keras
import innvestigate
import innvestigate.utils as iutils
from innvestigate.analyzer.base import AnalyzerBase

from XAIRT.utils import getLayerIndexByName 

from sklearn.linear_model import LinearRegression
from keras import Model
from beartype import beartype
from beartype.typing import Any, Dict, List, Optional, Tuple, Union
from jaxtyping import Float

__all__ = ["XLR", "XAIKeras"]

class XLR:
    """
    In an XAI context, only normalized samples makes sense for XLR
    Since all inputs should be of a similar scale to compare coeffs.
    """

    def __init__(self,
                 model: LinearRegression,
                 samples: Float[np.ndarray, "dimy dimx"],
                 normalize: Dict[str, Any] = {"bool_": True, "kind": "Sum"}
                 ) -> None:

        super().__init__()
        self.model = model
        self.samples = samples
        self.normalize = normalize
        self._coef = self.model.coef_
        self.fit_intercept = self.model.fit_intercept

    def _analyze_sample(self,
                        sample: Float[np.ndarray, "dimx"],
                        normalize: Dict[str, Any] = {"bool_": True, "kind": "Sum"}
                        ) -> Float[np.ndarray, "dimx"]:

        a = self._coef * sample

        if normalize["bool_"] is True and "kind" not in normalize:
            normalize["kind"] = "Sum"
        else:
            pass

        if normalize["bool_"] is True and normalize["kind"] == "MaxAbs":
            a /= np.nanmax(np.abs(a))
        elif normalize["bool_"] is True and normalize["kind"] == "Sum":
            a /= np.nansum(a)
        elif normalize["bool_"] is True and normalize["kind"] != "MaxAbs" and normalize["kind"] != "Sum":
            raise NotImplementedError("Only MaxAbs and Sum normalization currently available!")
        else:
            pass

        return a

    def analyze_samples(self,
                        samples: Float[np.ndarray, "dimy dimx"],
                        normalize: Dict[str, Any] = {"bool_": True, "kind": "Sum"}
                        ) -> Float[np.ndarray, "dimy dimx"]:

        a = np.zeros(samples.shape, dtype = np.float64)
        numSamples = samples.shape[0]

        for i in range(numSamples):
            a[i] = self._analyze_sample(samples[i], normalize)

        return a

    def quick_analyze(self) -> Tuple[Float[np.ndarray, "dimy dimx"], Dict[str, Float[np.ndarray, "..."]]]:

        a = self.analyze_samples(self.samples, self.normalize)
        statistics = self.compute_statistics(a)

        return a, statistics

    @staticmethod
    def compute_statistics(a: Float[np.ndarray, "dimy dimx"]) -> Dict[str, Float[np.ndarray, "..."]]:
        
        Stats = {}

        # Mean heatmap over all samples
        Stats["mean"] = np.nanmean(a, axis = 0)

        return Stats

class XAIKeras:

    def __init__(self, 
                 model: Model,
                 method: Dict[str, Any],
                 kind: str,
                 samples: Float[np.ndarray, "dimy dimx"],
                 normalize: Dict[str, Any] = {"bool_": True, "kind": "Sum"},
                 **kwargs: Dict[str, Any]) -> None:

        super().__init__()
        self.model = model
        self.method = method
        self.kind = kind
        self.samples = samples
        self.normalize = normalize
        self.kwargs = kwargs

    def _create_analyzer(self, 
                         method: Dict[str, Any],
                         kind: str,
                         sample: Float[np.ndarray, "dimx"],
                         **kwargs: Dict[str, Any]) -> AnalyzerBase:
        
        if kind == "classic":
            Analyze = innvestigate.create_analyzer(method["name"], self.model, **method["optParams"])
        else:
            raise NotImplementedError("The only kinds of analyzers available are classic!")

        return Analyze

    def _analyze_sample(self,
                        method: Dict[str, Any],
                        kind: str,
                        sample: Float[np.ndarray, "dimx"],
                        normalize: Dict[str, Any] = {"bool_": True, "kind": "Sum"},
                        Analyze: Optional[AnalyzerBase],
                        **kwargs: Dict[str, Any]
                        ) -> Float[np.ndarray, "dimx"]:

        if kind =="classic" and Analyze is not None:
            a = Analyze.analyze(sample[np.newaxis,:])
        elif kind == "classic" and Analyze is None:
            Analyze = self._create_analyzer(method, kind, sample, **kwargs)
            a = Analyze.analyze(sample[np.newaxis,:])
        else:
            raise NotImplementedError("The only kinds of analyzers available are classic!")

        if normalize["bool_"] is True and "kind" not in normalize:
            normalize["kind"] = "Sum"

        if normalize["bool_"] is True and normalize["kind"] == "MaxAbs":
            a /= np.nanmax(np.abs(a))
        elif normalize["bool_"] is True and normalize["kind"] == "Sum":
            a /= np.nansum(a)
        elif normalize["bool_"] is True and normalize["kind"] != "MaxAbs" and normalize["kind"] != "Sum":
            raise NotImplementedError("Only MaxAbs and Sum normalization currently available!")
        else:
            pass

        return a
        
    def analyze_samples(self,
                        method: Dict[str, Any],
                        kind: str,
                        samples: Float[np.ndarray, "dimy dimx"],
                        normalize: Dict[str, Any] = {"bool_": True, "kind": "Sum"},
                        Analyze: Optional[AnalyzerBase],
                        **kwargs: Dict[str, Any]
                        ) -> Float[np.ndarray, "dimy dimx"]:

        a = np.zeros(samples.shape, dtype = np.float64)
        numSamples = samples.shape[0]

        count_allZeros = 0
        for i in range(numSamples):
            a[i] = self._analyze_sample(method, kind, samples[i], normalize, Analyze, **kwargs)
            if np.nansum(a[i]) == 0:
                count_allZeros = count_allZeros + 1

        print(f"Number of all-zero samples detected : {count_allZeros} i.e. {count_allZeros*100.0/numSamples} %")

        return a

    def quick_analyze(self) -> Tuple[Float[np.ndarray, "dimy dimx"], Dict[str, Float[np.ndarray, "..."]]]:

        Analyze = self._create_analyzer(self.method, self.kind, **self.kwargs)
        a = self.analyze_samples(self.method, self.kind, self.samples, self.normalize, Analyze, **self.kwargs)
        statistics = self.compute_statistics(a)

        return a, statistics

    @staticmethod
    def compute_statistics(a: Float[np.ndarray, "dimy dimx"]) -> Dict[str, Float[np.ndarray, "..."]]:

        Stats = {}

        # Mean heatmap over all samples
        Stats["mean"] = np.nanmean(a, axis = 0)

        return Stats

