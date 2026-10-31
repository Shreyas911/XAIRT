import numpy as np

import torch
import torch.nn as nn
import innvestigate
from innvestigate.analyzer.base import AnalyzerBase
from captum.attr import IntegratedGradients, Saliency, DeepLift, InputXGradient, LRP
from captum.attr._utils.lrp_rules import EpsilonRule, Alpha1_Beta0_Rule, PropagationRule
from torch import Tensor
from typing import List, cast

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

_LRP_METHODS = ("lrp.z", "lrp.epsilon", "lrp.alpha_1_beta_0", "lrp.alpha_1_beta_0_IB")

class Alpha1_Beta0_SignSplit_Rule(Alpha1_Beta0_Rule):
    """
    innvestigate's A1B0 as a Captum rule: the positive contributions are (x_i w_ij)+, i.e. the positive weights
    on the positive inputs and the negative weights on the negative inputs. Captum's Alpha1_Beta0_Rule uses
    x_i (w_ij)+, which is the same only for inputs >= 0. The weights are chosen from the sign of each input,
    so this works for one sample at a time, which is how XAITorch calls Captum.
    """

    # pyre-fixme[2]: Parameter must be annotated.
    def _manipulate_weights(self, module, inputs, outputs) -> None:
        x = inputs[0]
        if x.shape[0] != 1:
            raise ValueError("Alpha1_Beta0_SignSplit_Rule chooses the weights from the sign of the input, so it needs a batch of one sample.")
        if hasattr(module, "weight"):
            w = module.weight.data    # (out, in)
            module.weight.data = torch.where(x[0] > 0, w.clamp(min=0), w.clamp(max=0))
        if self.set_bias_to_zero and hasattr(module, "bias"):
            if module.bias is not None:
                module.bias.data = torch.zeros_like(module.bias.data)

class BoundedRule(PropagationRule):
    """
    innvestigate's BoundedRule (the z^B rule of Deep Taylor Decomposition, Montavon et al. 2017) as a Captum rule, for
    the first layer, the inputs x_i in [low_i, high_i]. The bias is ignored, as in innvestigate:
        R_i = sum_j (x_i w_ij - low_i w_ij+ - high_i w_ij-) / sum_i' (x_i' w_i'j - low_i' w_i'j+ - high_i' w_i'j-) R_j
    low and high are scalars or arrays of one value per input. Captum computes the relevance of an input as x_i times
    the gradient through the layer. The weights are kept and the bias is replaced by -sum_i' (low_i' w_i'j+ + high_i' w_i'j-),
    which makes the output of the layer the denominator, and the input hook subtracts the low and high terms.
    """

    def __init__(self, low = -1.0, high = 1.0) -> None:
        self.low = low
        self.high = high

    # pyre-fixme[2]: Parameter must be annotated.
    def _manipulate_weights(self, module, inputs, outputs) -> None:
        if getattr(module, "bias", None) is None:
            raise NotImplementedError("BoundedRule replaces the bias of the layer, so the layer needs one.")
        w = module.weight.data    # (out, in), left as it is
        self._w_pos, self._w_neg = w.clamp(min=0), w.clamp(max=0)
        self._low = torch.broadcast_to(torch.as_tensor(self.low, dtype=w.dtype, device=w.device), (w.shape[1],))
        self._high = torch.broadcast_to(torch.as_tensor(self.high, dtype=w.dtype, device=w.device), (w.shape[1],))
        module.bias.data = -(self._w_pos @ self._low + self._w_neg @ self._high)

    # pyre-fixme[3]: Return type must be annotated.
    def _create_backward_hook_output(self, outputs: torch.Tensor):
        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def _backward_hook_output(grad):
            sign = torch.sign(outputs)
            sign[sign == 0] = 1
            relevance = grad / (outputs + sign * self.STABILITY_FACTOR)
            self.relevance_output[grad.device] = grad.data
            self._s = relevance    # R_j / z_j, for the input hook
            return relevance
        return _backward_hook_output

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def _create_backward_hook_input(self, inputs):
        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def _backward_hook_input(grad):
            # grad is sum_j w_ij s_j, the output hook ran first
            relevance = grad * inputs - self._low * (self._s @ self._w_pos) - self._high * (self._s @ self._w_neg)
            device = grad.device
            # pyre-fixme[16]: `PropagationRule` has no attribute `_has_single_input`.
            if self._has_single_input:
                self.relevance_input[device] = relevance.data
            else:
                cast(List[Tensor], self.relevance_input[device]).append(relevance.data)
            grad.replace_out = relevance
            return relevance
        return _backward_hook_input

class XAITorch(_XAIBase):
    """
    Same interface as XAIKeras, with Captum in place of innvestigate.

    The method dict uses the innvestigate names, so one dict works for both backends:
      'gradient', 'input_t_gradient', 'integrated_gradients', 'deep_lift.wrapper',
      'lrp.z', 'lrp.epsilon' (optParams: epsilon), 'lrp.alpha_1_beta_0',
      'lrp.alpha_1_beta_0_IB' (the bias is ignored, Captum's set_bias_to_zero). Both take
      optParams {'sign_split': True} for innvestigate's rule (Alpha1_Beta0_SignSplit_Rule).
    The LRP methods take optParams {'input_layer_rule': 'Bounded'} or {'input_layer_rule': (low, high)}
    for BoundedRule in the first layer, as innvestigate. Other names and optParams (e.g. the other input
    layer rules, bias, the other '_IB' variants) are innvestigate-specific and raise NotImplementedError.

    Captum's A1B0 clamps the weights to be non-negative and applies them to the
    input as it is. innvestigate splits the input into its positive and negative
    parts and uses the positive weights for the first and the negative weights for
    the second. The two are the same for inputs >= 0 (e.g. after a ReLU), and not
    for inputs of both signs, e.g. the first layer for anomalies.

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

            optParams = self._attach_lrp_rules(method)

            Analyze = LRP(self.model)

        else:
            raise NotImplementedError(f"Method '{name}' is not mapped. Available: "
                                      f"{list(_CAPTUM_METHODS) + list(_LRP_METHODS)}.")

        if optParams:
            raise NotImplementedError(f"optParams {list(optParams)} are not supported for '{name}' with Torch.")

        return Analyze

    @beartype
    def _attach_lrp_rules(self, method: Dict[str, Any]) -> Dict[str, Any]:
        """
        Puts the rule of an LRP method on the Linear layers, and returns the optParams that were not used.

        Captum deletes the rules of the model at the end of every attribute() call, and a layer without
        a rule gets the default epsilon rule, so this has to be done before each sample, not only when
        the analyzer is created. Otherwise every sample but the first is explained with LRP-Z.
        """

        name = method["name"]
        optParams = dict(method.get("optParams", {}))

        if name in ("lrp.alpha_1_beta_0", "lrp.alpha_1_beta_0_IB"):
            # sign_split: innvestigate's rule for inputs of both signs, see Alpha1_Beta0_SignSplit_Rule
            Rule = Alpha1_Beta0_SignSplit_Rule if optParams.pop("sign_split", False) else Alpha1_Beta0_Rule
            ignore_bias = name.endswith("_IB")
            make_rule = lambda: Rule(set_bias_to_zero = ignore_bias)
        elif name == "lrp.epsilon":
            epsilon = optParams.pop("epsilon", 1e-7)    # the innvestigate default
            make_rule = lambda: EpsilonRule(epsilon = epsilon)
        else:
            make_rule = EpsilonRule     # lrp.z, the default epsilon is negligible

        linears = [module for module in self.model.modules() if isinstance(module, nn.Linear)]
        for module in linears:
            module.rule = make_rule()

        # The rule of the first layer, as innvestigate's input_layer_rule: 'Bounded' (low -1, high 1) or (low, high)
        input_layer_rule = optParams.pop("input_layer_rule", None)
        if input_layer_rule == "Bounded":
            linears[0].rule = BoundedRule()
        elif isinstance(input_layer_rule, tuple):
            linears[0].rule = BoundedRule(*input_layer_rule)
        elif input_layer_rule is not None:
            raise NotImplementedError(f"input_layer_rule {input_layer_rule!r}: only 'Bounded' and (low, high) are available with Torch.")

        return optParams

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
        elif method["name"] in _LRP_METHODS:
            # Captum removed the rules after the previous sample
            self._attach_lrp_rules(method)

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
