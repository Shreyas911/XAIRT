import numpy as np

import torch
import torch.nn as nn
import innvestigate
from innvestigate.analyzer.base import AnalyzerBase
from captum.attr import IntegratedGradients, Saliency, DeepLift, InputXGradient, LRP
from captum.attr._utils.lrp_rules import EpsilonRule, Alpha1_Beta0_Rule

from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Model
from beartype import beartype
from beartype.typing import Any, Dict, Optional, Tuple
from jaxtyping import Float

__all__ = ["XLR", "XAIKeras", "XAITorch"]

class _XAIBase:
    """
    Normalization and statistics shared by XLR, XAIKeras and XAITorch, so that
    all three behave identically.

    Normalization divides without guarding against zero, so a degenerate sample
    (e.g. all-zero attributions) becomes NaN. This is deliberate: the nan-aware
    numpy functions used everywhere here (nanmean, nansum, nanmax) then ignore it.
    """

    @staticmethod
    @beartype
    def _apply_normalize(a: Float[np.ndarray, "..."],
                         normalize: Dict[str, Any]) -> Float[np.ndarray, "..."]:

        if not normalize.get("bool_", False):
            return a

        kind = normalize.get("kind", "Sum")
        if kind == "MaxAbs":
            return a / np.nanmax(np.abs(a))
        if kind == "Sum":
            return a / np.nansum(a)

        raise NotImplementedError("Only MaxAbs and Sum normalization currently available!")

    @staticmethod
    @beartype
    def _count_allZeros(a: Float[np.ndarray, "dimy dimx"]) -> None:

        numSamples = a.shape[0]
        count_allZeros = sum(np.nansum(np.abs(a[i])) == 0 for i in range(numSamples))

        print(f"Number of all-zero samples detected : {count_allZeros} i.e. {count_allZeros*100.0/numSamples} %")

    @staticmethod
    @beartype
    def compute_statistics(a: Float[np.ndarray, "dimy dimx"]) -> Dict[str, Float[np.ndarray, "..."]]:

        Stats = {}

        # Mean heatmap over all samples
        Stats["mean"] = np.nanmean(a, axis = 0)

        return Stats

class XLR(_XAIBase):
    """
    In an XAI context, only normalized samples makes sense for XLR
    Since all inputs should be of a similar scale to compare coeffs.
    """

    @beartype
    def __init__(self,
                 model: LinearRegression,
                 samples: Float[np.ndarray, "dimy dimx"],
                 normalize: Optional[Dict[str, Any]] = None
                 ) -> None:

        super().__init__()
        self.model = model
        self.samples = samples
        self.normalize = {"bool_": True, "kind": "Sum"} if normalize is None else normalize
        self._coef = self.model.coef_
        self.fit_intercept = self.model.fit_intercept

    @beartype
    def _analyze_sample(self,
                        sample: Float[np.ndarray, "dimx"],
                        normalize: Optional[Dict[str, Any]] = None
                        ) -> Float[np.ndarray, "dimx"]:

        a = self._coef * sample

        return self._apply_normalize(a, self.normalize if normalize is None else normalize)

    @beartype
    def analyze_samples(self,
                        samples: Float[np.ndarray, "dimy dimx"],
                        normalize: Optional[Dict[str, Any]] = None
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

class XAIKeras(_XAIBase):

    @beartype
    def __init__(self,
                 model: Model,
                 method: Dict[str, Any],
                 kind: str,
                 samples: Float[np.ndarray, "dimy dimx"],
                 normalize: Optional[Dict[str, Any]] = None,
                 **kwargs: Dict[str, Any]) -> None:

        super().__init__()
        self.model = model
        self.method = method
        self.kind = kind
        self.samples = samples
        self.normalize = {"bool_": True, "kind": "Sum"} if normalize is None else normalize
        self.kwargs = kwargs

    @beartype
    def _create_analyzer(self,
                         method: Dict[str, Any],
                         kind: str,
                         **kwargs: Dict[str, Any]) -> AnalyzerBase:

        if kind == "classic":
            Analyze = innvestigate.create_analyzer(method["name"], self.model, **method.get("optParams", {}))
        else:
            raise NotImplementedError("The only kinds of analyzers available are classic!")

        return Analyze

    @beartype
    def _analyze_sample(self,
                        method: Dict[str, Any],
                        kind: str,
                        sample: Float[np.ndarray, "dimx"],
                        normalize: Optional[Dict[str, Any]] = None,
                        Analyze: Optional[AnalyzerBase] = None,
                        **kwargs: Dict[str, Any]
                        ) -> Float[np.ndarray, "dimx"]:

        if kind != "classic":
            raise NotImplementedError("The only kinds of analyzers available are classic!")

        if Analyze is None:
            Analyze = self._create_analyzer(method, kind, **kwargs)

        # analyze() takes and returns a batch, here of one sample
        a = Analyze.analyze(sample[np.newaxis,:])[0]

        return self._apply_normalize(a, self.normalize if normalize is None else normalize)

    @beartype
    def analyze_samples(self,
                        method: Dict[str, Any],
                        kind: str,
                        samples: Float[np.ndarray, "dimy dimx"],
                        normalize: Optional[Dict[str, Any]] = None,
                        Analyze: Optional[AnalyzerBase] = None,
                        **kwargs: Dict[str, Any]
                        ) -> Float[np.ndarray, "dimy dimx"]:

        a = np.zeros(samples.shape, dtype = np.float64)
        numSamples = samples.shape[0]

        for i in range(numSamples):
            a[i] = self._analyze_sample(method, kind, samples[i], normalize, Analyze, **kwargs)

        self._count_allZeros(a)

        return a

    @beartype
    def quick_analyze(self) -> Tuple[Float[np.ndarray, "dimy dimx"], Dict[str, Float[np.ndarray, "..."]]]:

        Analyze = self._create_analyzer(self.method, self.kind, **self.kwargs)
        a = self.analyze_samples(self.method, self.kind, self.samples, self.normalize, Analyze, **self.kwargs)
        statistics = self.compute_statistics(a)

        return a, statistics

# innvestigate method name -> Captum attribution class. Method dicts are the same
# for both backends: dict(name='lrp.z', optParams={}, title='LRP-Z').
_CAPTUM_METHODS = {"gradient"            : Saliency,
                   "input_t_gradient"    : InputXGradient,
                   "integrated_gradients": IntegratedGradients,
                   "deep_lift.wrapper"   : DeepLift}

_LRP_METHODS = ("lrp.z", "lrp.epsilon", "lrp.alpha_1_beta_0")

class XAITorch(_XAIBase):
    """
    Same interface as XAIKeras, with Captum in place of innvestigate.

    The method dict uses the innvestigate names, so one dict works for both backends:
      'gradient', 'input_t_gradient', 'integrated_gradients', 'deep_lift.wrapper',
      'lrp.z', 'lrp.epsilon' (optParams: epsilon), 'lrp.alpha_1_beta_0'.
    Other names and optParams (e.g. input_layer_rule, bias, the '_IB' variants)
    are innvestigate-specific and raise NotImplementedError.

    Captum LRP does not support Softmax/Sigmoid layers, so for 'lrp.*' pass a model
    without its final one (the equivalent of innvestigate.model_wo_softmax, see
    model_wo_softmax_torch in XAIRT.utils).

    The model is put in eval mode, its ReLUs are made non-inplace (Captum needs
    that) and, for LRP, a `rule` is attached to its Linear layers.
    """

    @beartype
    def __init__(self,
                 model: nn.Module,
                 method: Dict[str, Any],
                 kind: str,
                 samples: Float[np.ndarray, "dimy dimx"],
                 normalize: Optional[Dict[str, Any]] = None,
                 **kwargs: Dict[str, Any]) -> None:

        super().__init__()
        self.model = model
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        self.model.eval()
        self.method = method
        self.kind = kind
        self.samples = samples
        self.normalize = {"bool_": True, "kind": "Sum"} if normalize is None else normalize
        self.kwargs = kwargs

    @beartype
    def _create_analyzer(self,
                         method: Dict[str, Any],
                         kind: str,
                         **kwargs: Dict[str, Any]) -> Any:

        if kind != "classic":
            raise NotImplementedError("The only kinds of analyzers available are classic!")

        name = method["name"]
        optParams = dict(method.get("optParams", {}))

        if name in _CAPTUM_METHODS:
            Analyze = _CAPTUM_METHODS[name](self.model)

        elif name in _LRP_METHODS:
            if any(isinstance(m, (nn.Softmax, nn.Sigmoid)) for m in self.model.modules()):
                raise ValueError("Captum LRP does not support Softmax/Sigmoid layers. "
                                 "Pass the model without its final one, see model_wo_softmax_torch.")

            # Set on every call, since rules stay attached to the model between analyzers
            if name == "lrp.alpha_1_beta_0":
                make_rule = Alpha1_Beta0_Rule
            elif name == "lrp.epsilon":
                epsilon = optParams.pop("epsilon", 1e-7)    # the innvestigate default
                make_rule = lambda: EpsilonRule(epsilon = epsilon)
            else:
                make_rule = EpsilonRule     # lrp.z, the default epsilon is negligible

            for module in self.model.modules():
                if isinstance(module, nn.Linear):
                    module.rule = make_rule()

            Analyze = LRP(self.model)

        else:
            raise NotImplementedError(f"Method '{name}' is not mapped. Available: "
                                      f"{list(_CAPTUM_METHODS) + list(_LRP_METHODS)}.")

        if optParams:
            raise NotImplementedError(f"optParams {list(optParams)} are not supported for '{name}' with Torch.")

        return Analyze

    @beartype
    def _get_target_idx(self, sample_t: torch.Tensor) -> int:
        """
        Predicted class index of a single-sample tensor, or 0 for a single output.
        Same as innvestigate's default neuron_selection, the max activation neuron.
        """

        with torch.no_grad():
            output = self.model(sample_t)

        if output.shape[-1] == 1:
            return 0

        return int(torch.argmax(output, dim=1).item())

    @beartype
    def _analyze_sample(self,
                        method: Dict[str, Any],
                        kind: str,
                        sample: Float[np.ndarray, "dimx"],
                        normalize: Optional[Dict[str, Any]] = None,
                        Analyze: Optional[Any] = None,
                        **kwargs: Dict[str, Any]
                        ) -> Float[np.ndarray, "dimx"]:

        if kind != "classic":
            raise NotImplementedError("The only kinds of analyzers available are classic!")

        if Analyze is None:
            Analyze = self._create_analyzer(method, kind, **kwargs)

        # Use the device the model is already on, as XAIKeras does not move anything
        device = next(self.model.parameters()).device
        sample_t = torch.from_numpy(sample.copy()).float().to(device).unsqueeze(0)
        target_idx = self._get_target_idx(sample_t)

        # Captum requires requires_grad=True on the input
        inp = sample_t.detach().requires_grad_(True)

        if isinstance(Analyze, Saliency):
            # Captum returns absolute values by default, innvestigate's gradient does not
            attr = Analyze.attribute(inp, target=target_idx, abs=False)
        else:
            attr = Analyze.attribute(inp, target=target_idx)

        a = attr.detach().cpu().numpy()[0]

        return self._apply_normalize(a, self.normalize if normalize is None else normalize)

    @beartype
    def analyze_samples(self,
                        method: Dict[str, Any],
                        kind: str,
                        samples: Float[np.ndarray, "dimy dimx"],
                        normalize: Optional[Dict[str, Any]] = None,
                        Analyze: Optional[Any] = None,
                        **kwargs: Dict[str, Any]
                        ) -> Float[np.ndarray, "dimy dimx"]:

        a = np.zeros(samples.shape, dtype = np.float64)
        numSamples = samples.shape[0]

        for i in range(numSamples):
            a[i] = self._analyze_sample(method, kind, samples[i], normalize, Analyze, **kwargs)

        self._count_allZeros(a)

        return a

    @beartype
    def quick_analyze(self) -> Tuple[Float[np.ndarray, "dimy dimx"], Dict[str, Float[np.ndarray, "..."]]]:

        Analyze = self._create_analyzer(self.method, self.kind, **self.kwargs)
        a = self.analyze_samples(self.method, self.kind, self.samples, self.normalize, Analyze, **self.kwargs)
        statistics = self.compute_statistics(a)

        return a, statistics
