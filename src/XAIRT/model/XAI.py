import torch
import torch.nn as nn
import numpy as np
import copy
from abc import ABCMeta, abstractmethod

from captum.attr import IntegratedGradients, Saliency, LRP, DeepLift

__all__ = ["X", "XLR", "XAI", "XAIR"]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class X(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self) -> None: pass

    @abstractmethod
    def _analyze_sample(self) -> np.ndarray: pass

    @abstractmethod
    def analyze_samples(self) -> np.ndarray: pass

    @abstractmethod
    def quick_analyze(self) -> tuple: pass

    @staticmethod
    def compute_statistics(a: np.ndarray) -> dict:
        return {'mean': np.nanmean(a, axis=0)}


# ---------------------------------------------------------------------------
# XLR  –  Linear Regression XAI
# ---------------------------------------------------------------------------

class XLR(X):
    """
    Linear attribution for sklearn LinearRegression models.
    Attribution = coef * (input - reference), where reference is zeros
    (only valid when fit_intercept=False).
    """

    def __init__(self, model, samples: np.ndarray = None,
                 normalize: dict = {'bool_': True, 'kind': 'Sum'}) -> None:
        super().__init__()
        self.model     = model
        self.samples   = samples
        self.normalize = normalize
        self._coef     = model.coef_
        self.fit_intercept = model.fit_intercept

        if self.fit_intercept:
            # With an intercept the zero-vector is not the natural reference;
            # the correct root is ambiguous, so we refuse early.
            raise NotImplementedError(
                "fit_intercept=True is not supported for XLR: the zero-vector "
                "is not a valid reference point when an intercept is present."
            )
        self.x_tilde = np.zeros(samples.shape[1])

    def _apply_normalize(self, a: np.ndarray, normalize: dict) -> np.ndarray:
        if normalize is None or not normalize.get('bool_', False):
            return a
        kind = normalize.get('kind', 'Sum')
        if kind == 'MaxAbs':
            denom = np.nanmax(np.abs(a))
            return a / denom if denom != 0 else a
        if kind == 'Sum':
            denom = np.nansum(a)
            return a / denom if denom != 0 else a
        raise NotImplementedError(f"Normalization kind '{kind}' not supported. "
                                  "Use 'MaxAbs' or 'Sum'.")

    def _analyze_sample(self, sample: np.ndarray,
                        normalize: dict = None) -> np.ndarray:
        a = self._coef * (sample - self.x_tilde)
        norm = normalize if normalize is not None else self.normalize
        return self._apply_normalize(a, norm)

    def analyze_samples(self, samples: np.ndarray = None,
                        normalize: dict = None) -> np.ndarray:
        S = samples if samples is not None else self.samples
        return np.array([self._analyze_sample(s, normalize) for s in S])

    def quick_analyze(self) -> tuple:
        if self.model is None or self.samples is None:
            raise ValueError("model and samples must both be set before "
                             "calling quick_analyze().")
        a = self.analyze_samples(self.samples, self.normalize)
        return a, self.compute_statistics(a)


# ---------------------------------------------------------------------------
# XAI  –  Neural-Network XAI via Captum
# ---------------------------------------------------------------------------

class XAI(X):
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

    def __init__(self, model: nn.Module,
                 method: dict = None,
                 kind: str = None,
                 samples: np.ndarray = None,
                 normalize: dict = {'bool_': True, 'kind': 'Sum'},
                 **kwargs) -> None:
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
        self.y_ref   = kwargs.get('y_ref', 0.0)
        self.kwargs  = kwargs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_analyzer(self, method_name: str):
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

    def _apply_normalize(self, a: np.ndarray, normalize: dict) -> np.ndarray:
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

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def _analyze_sample(self, sample: np.ndarray,
                        method: dict,
                        kind: str,
                        normalize: dict = None,
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

        elif kind == 'letzgus':
            models = self.createLetzgus(sample_t, self.y_ref)
            attrs = []
            for m in models:
                ig  = IntegratedGradients(m)
                inp = sample_t.detach().requires_grad_(True)
                ig_attr = ig.attribute(inp, target=target_idx)
                attrs.append(ig_attr.detach().cpu().numpy().flatten())
            a = np.sum(attrs, axis=0)

        else:
            raise NotImplementedError(
                f"kind='{kind}' is not supported. Use 'classic' or 'letzgus'."
            )

        norm = normalize if normalize is not None else self.normalize
        a = self._apply_normalize(a, norm)
        return a.reshape(sample.shape)

    def analyze_samples(self, samples: np.ndarray = None,
                        method: dict = None,
                        kind: str = None,
                        normalize: dict = None,
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

    def quick_analyze(self) -> tuple:
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

    # ------------------------------------------------------------------
    # Letzgus helpers
    # ------------------------------------------------------------------

    def _get_linear_layers_and_output(self):
        """
        Returns (partial_model, output_linear) where:
          - partial_model  : nn.Sequential of everything up to but NOT including
                             the last nn.Linear layer
          - output_linear  : the last nn.Linear layer object

        Works correctly for the architecture produced by TrainFullyConnectedNN:
          [Linear, ReLU, Linear, ReLU, Linear, Softmax]

        The key correctness constraint for Letzgus is that we need the
        pre-activation values feeding INTO the last Linear, so partial_model
        must stop right before the last Linear (not at the last activation).
        """
        all_children = list(self.model.children())

        # Find the last child that is an nn.Linear
        last_lin_idx = max(
            (i for i, c in enumerate(all_children) if isinstance(c, nn.Linear)),
            default=None
        )
        if last_lin_idx is None or last_lin_idx == 0:
            raise ValueError("Model must have at least 2 nn.Linear layers "
                             "for Letzgus analysis.")

        partial_model = nn.Sequential(*all_children[:last_lin_idx])
        output_linear = all_children[last_lin_idx]   # guaranteed nn.Linear

        if not isinstance(output_linear, nn.Linear):
            raise ValueError(
                f"Expected nn.Linear at child index {last_lin_idx}, "
                f"got {type(output_linear)}."
            )
        return partial_model, output_linear

    def offsetLetzgus(self, sample_t: torch.Tensor,
                      y_ref: float,
                      step_width: float = 0.00005,
                      max_it: int = 10_000) -> torch.Tensor:
        """
        Iteratively adjusts the penultimate-layer activation until the model
        output equals y_ref (flooding method). Returns the adjusted a_ref.

        Fixes vs uploaded version:
          - output_layer is always nn.Linear (not nn.Softmax)
          - current_y correctly indexes the target class for multi-output nets
          - target_idx is computed once and reused inside the loop
        """
        partial_model, output_linear = self._get_linear_layers_and_output()

        with torch.no_grad():
            a_ref = partial_model(sample_t.detach()).squeeze()  # (hidden_size,)

            # Get initial target class from full model
            target_idx = self._get_target_idx(sample_t)

            def _forward_last(a):
                """Compute output of the last linear for a given activation."""
                y_vec = torch.matmul(output_linear.weight, a)
                if output_linear.bias is not None:
                    y_vec = y_vec + output_linear.bias
                # For multi-output: return scalar for target class
                return y_vec[target_idx].item() if y_vec.numel() > 1 else y_vec.item()

            current_y = _forward_last(a_ref)
            update    = torch.full_like(a_ref, step_width)
            counter   = 0

            while abs(current_y - y_ref) > 1e-4 and counter < max_it:
                a_ref     = torch.clamp(a_ref + (-update if current_y >= y_ref else update), min=0.0)
                current_y = _forward_last(a_ref)
                counter  += 1
                print(f"iteration {counter:6d} | y: {current_y:.6f}", end='\r')

            if counter >= max_it:
                print(f"\n! y_ref={y_ref} not reached within {max_it} iterations "
                      f"(final y={current_y:.6f}).")
            else:
                print()   # newline after \r progress

        return a_ref

    def triplicateLetzgus(self, a_ref: torch.Tensor) -> list:
        """
        Builds the 3-model Letzgus ensemble by weight manipulation.

        Model semantics (mirrors original Keras version):
          m1: penultimate bias shifted down by a_ref
          m2: penultimate weights and bias negated
          m3: penultimate weights negated, bias = -bias + a_ref,
              output weights negated

        Fix vs uploaded version:
          - Layer assignment uses .data[:] = ... instead of direct
            parameter replacement (= ...), which would detach the
            tensor from the parameter graph and break state_dict().
        """
        if not hasattr(self, '_letzgus_models'):
            self._letzgus_models = [
                copy.deepcopy(self.model), 
                copy.deepcopy(self.model), 
                copy.deepcopy(self.model)
                ]

        m1, m2, m3 = self._letzgus_models
        def _get_linear_pair(m):
            # Helper to get the penultimate and final linear layers
            linears = [l for l in m.modules() if isinstance(l, nn.Linear)]
            if len(linears) < 2:
                raise ValueError("Model requires at least 2 Linear layers.")
            return linears[-2], linears[-1]

        with torch.no_grad():
            # Get layer references
            l1_pen, l1_out = _get_linear_pair(m1)
            l2_pen, l2_out = _get_linear_pair(m2)
            l3_pen, l3_out = _get_linear_pair(m3)
        
            # Original weights/biases for resetting (from self.model)
            orig_pen, orig_out = _get_linear_pair(self.model)

            # Model 1: Penultimate bias shifted by a_ref
            l1_pen.bias.copy_(orig_pen.bias - a_ref)
        
            # Model 2: Negate penultimate weights and bias
            l2_pen.weight.copy_(orig_pen.weight * -1)
            l2_pen.bias.copy_(orig_pen.bias * -1)

            # Model 3: Negate penultimate weight; bias = -orig_bias + a_ref; negate output weight
            l3_pen.weight.copy_(orig_pen.weight * -1)
            l3_pen.bias.copy_((orig_pen.bias * -1) + a_ref)
            l3_out.weight.copy_(orig_out.weight * -1)

        return [m1, m2, m3]

    def createLetzgus(self, sample_t: torch.Tensor, y_ref: float) -> list:
        a_ref = self.offsetLetzgus(sample_t, y_ref)
        return self.triplicateLetzgus(a_ref)

    def check_sample(self, sample: np.ndarray) -> float:
        """
        Sanity check: the sum of the three Letzgus model outputs should
        equal (model_output - y_ref). Returns the residual; should be ~0.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        sample_t = torch.from_numpy(sample.copy()).float().to(device).unsqueeze(0)
        with torch.no_grad():
            y_full = self.model(sample_t)
            target_idx = self._get_target_idx(sample_t)
            y = y_full[0, target_idx].item() if y_full.numel() > 1 else y_full.item()
        models = self.createLetzgus(sample_t, self.y_ref)
        with torch.no_grad():
            y_sum = sum(
                m(sample_t)[0, target_idx].item()
                if m(sample_t).numel() > 1 else m(sample_t).item()
                for m in models
            )
        return y - y_sum - self.y_ref


# Backwards-compatible alias
XAIR = XAI
