import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from abc import ABCMeta, abstractmethod
from sklearn.linear_model import LinearRegression

__all__ = ["Trainer", "TrainerNN", "TrainLR", "TrainFullyConnectedNN"]


# ---------------------------------------------------------------------------
# Abstract base classes  (mirrors original Keras hierarchy)
# ---------------------------------------------------------------------------

class Trainer(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self) -> None: pass
    @abstractmethod
    def _createModel(self) -> None: pass
    @abstractmethod
    def _trainModel(self) -> None: pass
    @abstractmethod
    def quickTrain(self): pass


class TrainerNN(Trainer):
    @abstractmethod
    def __init__(self) -> None: pass
    @abstractmethod
    def _compileModel(self) -> None: pass   # no-op in PyTorch; validates config
    @abstractmethod
    def _createCheckpoint(self) -> None: pass  # no-op; pre-creates output dir
    @abstractmethod
    def loadBestModel(self) -> nn.Module: pass


# ---------------------------------------------------------------------------
# TrainFullyConnectedNN
# ---------------------------------------------------------------------------

class TrainFullyConnectedNN(TrainerNN):
    """
    Fully-connected feed-forward network trainer (PyTorch backend).

    Parameters
    ----------
    x, y            : numpy arrays or torch Tensors (converted to float32)
    layers          : list of layer-config dicts.
                      First entry = input layer (size key only, no activation).
                      Example:
                        [{'size': 16},
                         {'size': 64, 'activation': 'relu', 'use_bias': True,
                          'l1_w_reg': 0.0, 'l2_w_reg': 1e-4,
                          'l1_b_reg': 0.0, 'l2_b_reg': 0.0},
                         {'size':  2, 'activation': 'softmax', 'use_bias': True}]
    losses          : [{'kind': 'crossentropy', 'weight': 1.0}]
                      Supported: crossentropy / categorical_crossentropy,
                      mse, mae, bce, bce_with_logits.
    optimizer       : 'adam' | 'sgd'
    metrics         : list[str]  – informational only, not used during training
    batch_size      : int
    epochs          : int
    filename        : checkpoint filename stem  (<filename>.pt)
    dirname         : directory for checkpoint files
    verbose         : 0 = silent | 1 = print every 10 epochs
    validation_data : (x_val, y_val) numpy arrays or Tensors
    random_nn_seed  : int | None  – seeds torch before weight init
    decay_rate      : float | None  – ExponentialLR gamma (ignored when
                      decay_func is supplied to quickTrain)
    class_weight    : dict {class_idx: float}  – per-class loss multipliers,
                      matching Keras class_weight semantics
    """

    def __init__(self,
                 x, y,
                 layers: list,
                 losses: list,
                 optimizer: str,
                 metrics: list,
                 batch_size: int,
                 epochs: int,
                 filename: str,
                 dirname: str,
                 verbose: int = 1,
                 validation_data: tuple = None,
                 random_nn_seed: int = None,
                 decay_rate: float = None,
                 **kwargs) -> None:

        super().__init__()

        def _to_tensor(arr):
            """Convert numpy → contiguous float32 tensor, or ensure float32."""
            if isinstance(arr, np.ndarray):
                # .copy() ensures C-contiguous layout; required by from_numpy
                return torch.from_numpy(arr.copy()).float()
            return arr.float() if arr.dtype != torch.float32 else arr

        self.x = _to_tensor(x)
        self.y = _to_tensor(y)

        self.layers_cfg = layers
        self.losses_cfg = losses
        self.opt_name   = optimizer.lower()
        self.decay_rate = decay_rate
        self.metrics    = metrics
        self.verbose    = verbose
        self.batch_size = batch_size
        self.epochs     = epochs
        self.dirname    = dirname
        self.filename   = filename

        # Seed before _createModel so weight init is reproducible
        self.random_nn_seed = random_nn_seed
        if self.random_nn_seed is not None:
            torch.manual_seed(self.random_nn_seed)

        # decay_func: callable (epoch -> lr), set via quickTrain().
        # Direct param_group assignment mirrors Keras LearningRateScheduler
        # exactly and avoids the LambdaLR step-counter ≠ epoch-index mismatch
        # that caused explosive loss (e.g. 0.75 -> 1504) for some lags.
        self.decay_func = None

        # class_weight: dict {class_idx: float}
        self.class_weight = kwargs.get('class_weight', None)

        if validation_data is not None:
            vx, vy = validation_data
            self.validation_data = (_to_tensor(vx), _to_tensor(vy))
        else:
            self.validation_data = None

        self.mod_path      = os.path.join(self.dirname, self.filename + '.pt')
        os.makedirs(self.dirname, exist_ok=True)

        self.best_val_loss = float('inf')
        self._model_state  = []
        self.model         = None

        self.model_metadata = {
            'layers': self.layers_cfg, 'losses': self.losses_cfg,
            'optimizer': self.opt_name, 'decay_rate': self.decay_rate,
            'metrics': self.metrics,
        }
        self.train_metadata = {
            'batch_size': self.batch_size, 'epochs': self.epochs,
            'validation_data': self.validation_data,
            'filename': self.filename, 'dirname': self.dirname,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_activation(name):
        """Return a fresh activation Module for the given name, or None."""
        if name is None:
            return None
        name = name.lower()
        if name == 'relu':        return nn.ReLU()
        if name == 'sigmoid':     return nn.Sigmoid()
        if name == 'tanh':        return nn.Tanh()
        if name == 'softmax':     return nn.Softmax(dim=1)
        if name == 'leaky_relu':  return nn.LeakyReLU()
        raise NotImplementedError(f"Activation '{name}' is not supported.")

    @staticmethod
    def _safe_use_bias(cfg) -> bool:
        """
        Extract use_bias from a layer config dict.
        The input-layer sentinel value is None — treat that as True
        (bias flag only matters for nn.Linear layers anyway).
        """
        val = cfg.get('use_bias', True)
        return True if val is None else bool(val)

    def _build_model_modules(self) -> list:
        """
        Build the list of nn.Modules from self.layers_cfg.
        Shared by _createModel and loadBestModel so architecture is
        defined in exactly one place.
        """
        modules = []
        for i in range(1, len(self.layers_cfg)):
            cfg      = self.layers_cfg[i]
            in_dim   = self.layers_cfg[i - 1]['size']
            out_dim  = cfg['size']
            use_bias = self._safe_use_bias(cfg)

            linear = nn.Linear(in_dim, out_dim, bias=use_bias)
            # Kaiming-Normal init matches Keras HeNormal initializer
            nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
            if use_bias:
                nn.init.zeros_(linear.bias)
            modules.append(linear)

            act = self._get_activation(cfg.get('activation'))
            if act is not None:
                modules.append(act)
        return modules

    def _build_criterion(self):
        """
        Returns (criterion_fn, loss_kind, has_softmax_output).

        Loss selection rationale:
          nn.CrossEntropyLoss internally applies log_softmax.  When the model
          already ends with nn.Softmax, feeding its output to CrossEntropyLoss
          computes log(softmax(softmax(x))) — a double-softmax that drives
          val_loss → ln(2) ≈ 0.693 (random-output baseline) and blocks learning.

          Keras categorical_crossentropy EXPECTS softmax outputs (it just takes
          -sum(y * log(p))).  The PyTorch equivalent for softmax outputs against
          one-hot float targets is nn.BCELoss, which is what we use here.

        Regularisation note (in _trainModel, not here):
          Keras L1L2 penalty = l1*sum(|w|) + 0.5*l2*sum(w²).
          torch.norm(w,2) = sqrt(sum(w²))  ← completely wrong scale and gradient.
          We use (w**2).sum() with the 0.5 factor to match Keras exactly.
        """
        if len(self.losses_cfg) != 1:
            raise NotImplementedError("Multiple/weighted losses not implemented.")
        if self.losses_cfg[0].get('weight', 1.0) != 1.0:
            raise ValueError("Loss weight must be 1.0 for a single loss.")

        kind     = self.losses_cfg[0]['kind'].lower()
        last_act = self.layers_cfg[-1].get('activation', None)
        has_softmax = (last_act is not None and last_act.lower() == 'softmax')

        # reduction='none' so we can apply per-sample class weights before .mean()
        if kind in ('crossentropy', 'categorical_crossentropy'):
            crit = nn.BCELoss(reduction='none') if has_softmax \
                   else nn.CrossEntropyLoss(reduction='none')
        elif kind == 'mse':
            crit = nn.MSELoss(reduction='none')
        elif kind == 'mae':
            crit = nn.L1Loss(reduction='none')
        elif kind == 'bce':
            crit = nn.BCELoss(reduction='none')
        elif kind == 'bce_with_logits':
            crit = nn.BCEWithLogitsLoss(reduction='none')
        else:
            raise NotImplementedError(f"Loss '{kind}' not supported.")

        return crit, kind, has_softmax

    def _build_optimizer(self) -> optim.Optimizer:
        params = self.model.parameters()
        if self.opt_name == 'adam':
            return optim.Adam(params, lr=1e-3)
        if self.opt_name == 'sgd':
            # nesterov=True matches keras.optimizers.SGD(nesterov=True)
            return optim.SGD(params, lr=1e-2, momentum=0.9, nesterov=True)
        raise NotImplementedError(f"Optimizer '{self.opt_name}' not supported.")

    # ------------------------------------------------------------------
    # TrainerNN abstract implementations
    # ------------------------------------------------------------------

    def _compileModel(self) -> None:
        """Validates loss/optimizer config early; no-op otherwise."""
        self._build_criterion()   # raises on bad config
        self._model_state.append('compiled')

    def _createCheckpoint(self) -> None:
        """Pre-creates output directory so _trainModel can save without checks."""
        os.makedirs(self.dirname, exist_ok=True)
        self._model_state.append('checkpointed')

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def _createModel(self) -> None:
        self.model = nn.Sequential(*self._build_model_modules())
        self._model_state.append('created')

    def _trainModel(self) -> None:
        # 1. Device Discovery
        device = torch.device("cuda" if torch.cuda.is_available() else 
                              "mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(device)

        dataset = TensorDataset(self.x, self.y)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 2. Setup Logic
        crit, loss_kind, has_softmax = self._build_criterion()
        optimizer = self._build_optimizer()

        # Handle Class Weights
        cw_tensor = None
        if self.class_weight is not None:
            n_cls = self.layers_cfg[-1]['size']
            cw_tensor = torch.tensor(
                [self.class_weight.get(i, 1.0) for i in range(n_cls)],
                dtype=torch.float32
            ).to(device)

        for epoch in range(self.epochs):
            # 3. Learning Rate Scheduler (Keras-style)
            if self.decay_func:
                new_lr = float(self.decay_func(epoch))
                for pg in optimizer.param_groups:
                    pg['lr'] = new_lr

            self.model.train()
            running_loss = 0.0
            
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                
                # 4. Loss Calculation (Handling Logits vs Softmax)
                if loss_kind in ('crossentropy', 'categorical_crossentropy') and not has_softmax:
                    targets  = batch_y.argmax(dim=1).long()
                    raw_loss = crit(outputs, targets)
                    if cw_tensor is not None:
                        raw_loss = raw_loss * cw_tensor[targets]
                    loss = raw_loss.mean()
                else:
                    raw_loss = crit(outputs, batch_y)
                    if cw_tensor is not None:
                        raw_loss = raw_loss * cw_tensor.unsqueeze(0)
                    loss = raw_loss.mean()

                # 5. Regularization (Matches Keras L1L2 exactly)
                reg = torch.zeros(1, device=device)
                lin_idx = 1
                for layer in self.model:
                    if isinstance(layer, nn.Linear):
                        if lin_idx < len(self.layers_cfg):
                            cfg = self.layers_cfg[lin_idx]
                            l1, l2 = cfg.get('l1_w_reg', 0.0) or 0.0, cfg.get('l2_w_reg', 0.0) or 0.0
                            l1_b, l2_b = cfg.get('l1_b_reg', 0.0) or 0.0, cfg.get('l2_b_reg', 0.0) or 0.0
                            if l1: reg += l1 * layer.weight.abs().sum()
                            if l2: reg += 0.5 * l2 * (layer.weight ** 2).sum()
                            if l1_b and layer.bias is not None: reg += l1_b * layer.bias.abs().sum()
                            if l2_b and layer.bias is not None: reg += 0.5 * l2_b * (layer.bias ** 2).sum()
                        lin_idx += 1

                total_loss = loss + reg.squeeze()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += total_loss.item()

            # 6. Validation and Checkpointing
            epoch_loss = running_loss / len(loader)
            if self.validation_data is not None:
                self.model.eval()
                with torch.no_grad():
                    val_x, val_y = self.validation_data[0].to(device), self.validation_data[1].to(device)
                    val_out = self.model(val_x)
                    if loss_kind in ('crossentropy', 'categorical_crossentropy') and not has_softmax:
                        vt = val_y.argmax(dim=1).long()
                        monitor_loss = crit(val_out, vt).mean().item()
                    else:
                        monitor_loss = crit(val_out, val_y).mean().item()
            else:
                monitor_loss = epoch_loss

            if monitor_loss < self.best_val_loss:
                self.best_val_loss = monitor_loss
                torch.save(self.model.state_dict(), self.mod_path)

            if self.verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:4d} | train loss: {epoch_loss:.6f} | val loss: {monitor_loss:.6f}")

        self._model_state.append('trained')

    def loadBestModel(self) -> nn.Module:
        """
        Builds a fresh model, loads the best checkpoint weights into it,
        and returns it — without touching self.model.
        """
        if not self._model_state or self._model_state[-1] != 'trained':
            raise RuntimeError("Model has not been trained yet.")

        # Rebuild architecture (init weights are immediately overwritten)
        best_model = nn.Sequential(*self._build_model_modules())

        # weights_only=True suppresses FutureWarning on PyTorch >= 2.0
        # and is required in future PyTorch versions
        try:
            state = torch.load(self.mod_path, map_location='cpu',
                               weights_only=True)
        except TypeError:
            # PyTorch < 1.13 does not have the weights_only argument
            state = torch.load(self.mod_path, map_location='cpu')

        best_model.load_state_dict(state)
        best_model.eval()
        return best_model

    def quickTrain(self, decay_func=None) -> nn.Module:
        """
        Full pipeline: create ->compile ->checkpoint ->train ->return best.

        Parameters
        ----------
        decay_func : callable (epoch -> lr) | None
            Mirrors Keras LearningRateScheduler; applied by direct
            param_group assignment at the start of each epoch.
        """
        self._model_state  = []
        self.best_val_loss = float('inf')
        self.decay_func    = decay_func

        self._createModel()
        self._compileModel()
        self._createCheckpoint()
        self._trainModel()

        return self.loadBestModel()


# ---------------------------------------------------------------------------
# TrainLR
# ---------------------------------------------------------------------------

class TrainLR(Trainer):
    """
    Linear Regression trainer (scikit-learn, closed-form solution).
    """

    def __init__(self, x: np.ndarray, y: np.ndarray,
                 fit_intercept: bool = False,
                 y_ref: float = 0.0) -> None:
        super().__init__()
        self.x             = x
        self.y             = y
        self.fit_intercept = fit_intercept
        self.y_ref         = y_ref
        self._model_state  = []

    def _createModel(self) -> None:
        self.regr = LinearRegression(fit_intercept=self.fit_intercept)
        self._model_state.append('created')

    def _trainModel(self) -> None:
        # Fit on (y - y_ref) — matches original Keras TrainLR
        self.regr.fit(self.x, self.y - self.y_ref)
        self._model_state.append('trained')

    def quickTrain(self) -> LinearRegression:
        self._model_state = []
        self._createModel()
        self._trainModel()
        return self.regr
