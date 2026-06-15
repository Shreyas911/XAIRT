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

    @beartype
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

    @beartype
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

    @beartype
    def analyze_samples(self,
                        samples: Float[np.ndarray, "dimy dimx"],
                        normalize: Dict[str, Any] = {"bool_": True, "kind": "Sum"}
                        ) -> Float[np.ndarray, "dimy dimx"]:

        a = np.zeros(samples.shape, dtype = np.float64)
        numSamples = samples.shape[0]

        for i in range(numSamples):
            a[i] = self._analyze_sample(samples[i], normalize)

        return a

    @beartype
    def quick_analyze(self) -> Tuple[Float[np.ndarray, "dimy dimx"], Dict[str, Float[np.ndarray, "..."]]]:

        a = self.analyze_samples(self.samples, self.normalize)
        statistics = self.compute_statistics(a)

        return a, statistics

    @beartype
    @staticmethod
    def compute_statistics(a: Float[np.ndarray, "dimy dimx"]) -> Dict[str, Float[np.ndarray, "..."]]:
        
        Stats = {}

        # Mean heatmap over all samples
        Stats["mean"] = np.nanmean(a, axis = 0)

        return Stats

class XAIKeras:

    @beartype
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

    @beartype
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

    @beartype
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

    @beartype
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

    @beartype
    def quick_analyze(self) -> Tuple[Float[np.ndarray, "dimy dimx"], Dict[str, Float[np.ndarray, "..."]]]:

        Analyze = self._create_analyzer(self.method, self.kind, **self.kwargs)
        a = self.analyze_samples(self.method, self.kind, self.samples, self.normalize, Analyze, **self.kwargs)
        statistics = self.compute_statistics(a)

        return a, statistics

    @beartype
    @staticmethod
    def compute_statistics(a: Float[np.ndarray, "dimy dimx"]) -> Dict[str, Float[np.ndarray, "..."]]:

        Stats = {}

        # Mean heatmap over all samples
        Stats["mean"] = np.nanmean(a, axis = 0)

        return Stats

class XAITorch:
    """
    Neural-network XAI using Captum as the attribution backend.
    Mirrors the original Keras XAIR interface.

    Parameters
    ----------
    model     : nn.Module  – trained PyTorch model (eval mode set automatically)
    method    : dict       – {'name': str, 'optParams': dict}
                             name: 'saliency' | 'integrated_gradients' |
                                   'deeplift' | 'lrp'
    kind      : str        – 'classic' | 'letzgus'
    samples   : np.ndarray – shape (N, n_features)
    normalize : dict       – {'bool_': bool, 'kind': 'Sum'|'MaxAbs'}
    y_ref     : float      – reference output for Letzgus (passed as kwarg)
    """

    @beartype
    def __init__(self,
                 model: nn.Module,
                 method: Dict[str, Any],
                 kind: str,
                 samples: Float[np.ndarray, "dimy dimx"],
                 normalize: Dict[str, Any] = {'bool_': True, 'kind': 'Sum'},
                 **kwargs: Dict[str, Any]) -> None:

        super().__init__()
        self.model   = model
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = False
        self.model.eval()
        self.method  = method
        self.kind    = kind
        self.samples = samples
        self.normalize = normalize
        self.kwargs  = kwargs

    @beartype
    def _get_analyzer(self, method_name: str) -> Any:

        """Maps method name string to a Captum attribution object."""
        name = method_name.lower()
        if name == 'saliency':            return Saliency(self.model)
        if name == 'integrated_gradients': return IntegratedGradients(self.model)
        if name == 'deeplift':            return DeepLift(self.model)
        if name == 'lrp':                 return LRP(self.model)

        raise NotImplementedError(
            f"Method '{method_name}' is not mapped. "
            "Available: saliency, integrated_gradients, deeplift, lrp."
        )

    @beartype
    def _apply_normalize(self, 
                         a: np.ndarray, 
                         normalize: Dict[str, Any] = {'bool_': True, 'kind': 'Sum'}) -> np.ndarray:

        if normalize is None or not normalize.get('bool_', False):
            return a

        kind = normalize.get('kind', 'Sum')
        if kind == 'MaxAbs':
            denom = np.nanmax(np.abs(a))
            return a / denom if denom != 0 else a
        if kind == 'Sum':
            denom = np.nansum(a)
            return a / denom if denom != 0 else a

        raise NotImplementedError(f"Normalization kind '{kind}' not supported.")

    @beartype
    def _get_target_idx(self, sample_t: torch.Tensor) -> int:
        """
        Returns the predicted class index for a single-sample tensor.
        For scalar regression output, returns 0.
        """
        with torch.no_grad():
            output = self.model(sample_t)
        if output.shape[-1] == 1:
            return 0
        return int(torch.argmax(output, dim=1).item())

    @beartype
    def _analyze_sample(self, 
                        sample: np.ndarray,
                        method: dict,
                        kind: str,
                        normalize: Dict[str, Any] = None,
                        **kwargs) -> np.ndarray:
        """
        Attribute a single sample. Returns an array of the same shape.
        """
        # Fresh tensor with grad enabled for Captum
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        sample_t = torch.from_numpy(sample.copy()).float().to(device).unsqueeze(0)
        target_idx = self._get_target_idx(sample_t)

        if kind == 'classic':
            analyzer = self._get_analyzer(method['name'])
            # Captum requires requires_grad=True on the input
            inp = sample_t.detach().requires_grad_(True)
            attr = analyzer.attribute(inp, target=target_idx)
            a = attr.detach().cpu().numpy().flatten()

        else:
            raise NotImplementedError(
                f"kind='{kind}' is not supported. Use 'classic' or 'letzgus'."
            )

        norm = normalize if normalize is not None else self.normalize
        a = self._apply_normalize(a, norm)
        return a.reshape(sample.shape)

    @beartype
    def analyze_samples(self, samples: np.ndarray = None,
                        method: dict = None,
                        kind: str = None,
                        normalize: Dict[str, Any] = None,
                        **kwargs) -> np.ndarray:
        """
        Attribute all samples. Returns array of same shape as samples.
        Counts and reports all-zero attribution vectors (sign of a bug).
        """
        S   = samples if samples is not None else self.samples
        m   = method  if method  is not None else self.method
        k   = kind    if kind    is not None else self.kind
        nrm = normalize if normalize is not None else self.normalize

        a = np.zeros(S.shape, dtype=np.float64)
        n_zero = 0
        for i in range(len(S)):
            a[i] = self._analyze_sample(S[i], m, k, nrm, **kwargs)
            if np.nansum(np.abs(a[i])) == 0:
                n_zero += 1

        print(f"All-zero attribution vectors: {n_zero} / {len(S)} "
              f"({100.0 * n_zero / len(S):.1f}%)")
        return a

    @beartype
    def quick_analyze(self) -> Tuple:
        """Full pipeline: attribute self.samples, return (attributions, stats)."""
        missing = [name for name, val in [
            ('model',   self.model),
            ('method',  self.method),
            ('kind',    self.kind),
            ('samples', self.samples),
        ] if val is None]
        if missing:
            raise ValueError(
                f"Cannot run quick_analyze(): {missing} are not set."
            )
        a = self.analyze_samples()
        return a, self.compute_statistics(a)

