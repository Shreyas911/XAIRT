import os
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras import initializers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.regularizers import L1L2

from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression

from tf import Tensor
from keras import Model

from beartype import beartype
from beartype.typing import Any, Dict, List, Optional, Tuple, Union
from jaxtyping import Float
from collections.abc import Callable

__all__ = ["TrainLR", "TrainKerasFullyConnectedNN", "TrainTorchFullyConnectedNN"]

class TrainLR:
    """
    In an XAI context, only a normalized x and y make sense for XLR.
    Since all inputs should be of a similar scale to compare coeffs.
    """

    @beartype
    def __init__(self,
                 x: Float[np.ndarray, "dimy dimx"],
                 y: Float[np.ndarray, "dimy"],
                 fit_intercept: bool = False,
                 y_ref: float = 0.0) -> None:

        super().__init__()

        self.x = x
        self.y = y
        self.fit_intercept = fit_intercept
        self.y_ref = y_ref
        self._model_state = []

    @beartype
    def train(self) -> LinearRegression:

        self.regr = LinearRegression(fit_intercept = self.fit_intercept)
        self.regr.fit(self.x, self.y-self.y_ref)
        return self.regr

class TrainKerasFullyConnectedNN:

    @beartype
    def __init__(self,
                 x: Float[np.ndarray, "dimy dimx"],
                 y: Float[np.ndarray, "dimy"],
                 layers: list[Dict[str, Any]],
                 losses: list[Dict[str, Any]],
                 optimizer: keras.optimizers.Optimizer,
                 metrics: list[str],
                 batch_size: int,
                 epochs: int,
                 filename: str,
                 dirname: str,
                 verbose: int = 1,
                 validation_split: Optional[float] = None,
                 validation_data: Optional[Tuple] = None,
                 class_weight: Optional[Dict[int, float]] = None,
                 random_nn_seed: Optional[int] = None,
                 decay_rate: Optional[float] = None,
                 custom_objects: Optional[Dict] = None) -> None:

        super().__init__()

        self.x = x
        self.y = y

        self.layers = layers

        self.losses = losses
        self.optimizer = optimizer
        self.decay_rate = decay_rate
        self.metrics = metrics
        self.verbose = verbose

        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.validation_data = validation_data
        self.class_weight = class_weight

        self.dirname = dirname
        self.filename = filename

        self.random_nn_seed = random_nn_seed

        self.model_metadata = {"layers"     : self.layers,
                               "losses"     : self.losses,
                               "optimizer"  : self.optimizer,
                               "decay_rate" : self.decay_rate,
                               "metrics"    : self.metrics}
        self.train_metadata = {"batch_size"       : self.batch_size,
                               "epochs"           : self.epochs,
                               "validation_split" : self.validation_split,
                               "filename"         : self.filename,
                               "dirname"          : self.dirname}
        self.callbacks = []
        self.custom_objects = custom_objects

    @beartype
    def _createModel(self) -> None:

        keras.backend.clear_session()

        _sizes = [layer["size"] for layer in self.layers]
        _activations = [layer["activation"] for layer in self.layers]
        _use_biases = [layer["use_bias"] if "use_bias" in layer else None for layer in self.layers]
        _l1_w_regs = [layer["l1_w_reg"] if "l1_w_reg" in layer else 0.0 for layer in self.layers]
        _l1_b_regs = [layer["l1_b_reg"] if "l1_b_reg" in layer else 0.0 for layer in self.layers]
        _l2_w_regs = [layer["l2_w_reg"] if "l2_w_reg" in layer else 0.0 for layer in self.layers]
        _l2_b_regs = [layer["l2_b_reg"] if "l2_b_reg" in layer else 0.0 for layer in self.layers]
        _kernel_constraints = [layer["kernel_constraint"] if "kernel_constraint" in layer else None for layer in self.layers]
        _bias_constraints = [layer["bias_constraint"] if "bias_constraint" in layer else None for layer in self.layers]

        if _activations[0] is not None:
            raise ValueError("Input layer cannot have an activation")
        if _use_biases[0] is not None:
            raise ValueError("Input layer cannot have a bias.")

        inputs = Input(shape=(_sizes[0],))
        dense = Dense(_sizes[1], 
                      activation=_activations[1], use_bias = _use_biases[1],
                      kernel_initializer=initializers.HeNormal(seed=self.random_nn_seed),
                      bias_initializer=initializers.HeNormal(seed=self.random_nn_seed),
                      kernel_regularizer=L1L2(l1 =_l1_w_regs[1], l2 = _l2_w_regs[1]),
                      bias_regularizer=L1L2(l1 = _l1_b_regs[1], l2 = _l2_b_regs[1]),
                      kernel_constraint=_kernel_constraints[1],
                      bias_constraint=_bias_constraints[1])
        x = dense(inputs)

        for i in range(2, len(_sizes)):

            dense = Dense(_sizes[i], 
                          activation=_activations[i], use_bias = _use_biases[i],
                          kernel_initializer=initializers.HeNormal(seed=self.random_nn_seed),
                          bias_initializer=initializers.HeNormal(seed=self.random_nn_seed),
                          kernel_regularizer=L1L2(l1 =_l1_w_regs[i], l2 = _l2_w_regs[i]),
                          bias_regularizer=L1L2(l1 = _l1_b_regs[i], l2 = _l2_b_regs[i]),
                          kernel_constraint=_kernel_constraints[i],
                          bias_constraint=_bias_constraints[i])
            x = dense(x)

        self.model = Model(inputs=inputs, outputs=x)

    @beartype
    def _compileModel(self) -> None:

        if len(self.losses) != 1:
            raise NotImplementedError("Weighted losses not implemented yet.")
        elif len(self.losses) == 1 and self.losses[0]["weight"] != 1.0:
            raise ValueError("Loss weight has to be 1.0 for a single loss function.")
        else: 
            pass

        _loss_kinds = [loss["kind"] for loss in self.losses]
        _loss_weights = [loss["weight"] for loss in self.losses]

        self.model.compile(loss=_loss_kinds[0], optimizer=self.optimizer, 
                           metrics=self.metrics)

    @beartype
    def _lrateSchedule(self, decay_func: Callable) -> None:

        self.lrate = LearningRateScheduler(decay_func)
        self.callbacks.append(self.lrate)

    @beartype
    def _createCheckpoint(self) -> None:

        self.mod_h5 = os.path.join(self.dirname,
                                   self.filename + ".h5")
        self.mod_txt  = os.path.join(self.dirname,
                                     self.filename + ".txt")
        self.checkpoint = ModelCheckpoint(self.mod_h5, monitor="val_loss",
                                           verbose=1,save_best_only=True)
        self.callbacks.append(self.checkpoint)

    @beartype
    def _trainModel(self) -> None:

        self._fit = self.model.fit(self.x, self.y,
                                    batch_size = self.batch_size,
                                    epochs = self.epochs,
                                    shuffle = True,
                                    validation_split = self.validation_split,
                                    validation_data = self.validation_data,
                                    callbacks = self.callbacks,
                                    class_weight = self.class_weight,
                                    verbose = self.verbose)

    @beartype
    def loadBestModel(self) -> Model:

        if self._model_state[-1] != "trained":
            raise Exception("Model is not trained!")

        best_model = keras.models.load_model(self.mod_h5, custom_objects=self.custom_objects)

        return best_model

    @beartype
    def train(self, decay_func: Optional[Callable] = None) -> Model:

        self._model_state = []

        self._createModel()
        self._compileModel()

        if decay_func is not None:
            self._lrateSchedule(decay_func)

        self._createCheckpoint()
        self._trainModel()

        return self.loadBestModel()

class TrainTorchFullyConnectedNN:

    @beartype
    def __init__(self,
                 x: Float[torch.Tensor, "dimy dimx"],
                 y: Float[torch.Tensor, "dimy"],
                 layers: list[Dict[str, Any]],
                 losses: list[Dict[str, Any]],
                 optimizer: str,
                 metrics: list[str],
                 batch_size: int,
                 epochs: int,
                 filename: str,
                 dirname: str,
                 verbose: int = 1,
                 validation_split: Optional[float] = None,
                 validation_data: Optional[Tuple] = None,
                 class_weight: Optional[Dict[int, float]] = None,
                 random_nn_seed: Optional[int] = None,
                 decay_rate: Optional[float] = None,
                 **kwargs: Dict[str, Any]) -> None:

        super().__init__()

        @beartype
        def _to_tensor(arr: Float[np.ndarray, ...]) -> Float[torch.Tensor, ...]:
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
        self.model         = None

        self.model_metadata = {"layers"     : self.layers_cfg,
                               "losses"     : self.losses_cfg,
                               "optimizer"  : self.opt_name,
                               "decay_rate" : self.decay_rate,
                               "metrics"    : self.metrics}
        self.train_metadata = {"batch_size"      : self.batch_size,
                               "epochs"          : self.epochs,
                               "validation_data" : self.validation_data,
                               "filename"        : self.filename, 
                               "dirname"         : self.dirname}

    @beartype
    @staticmethod
    def _get_activation(name) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:

        if name is None:
            return None

        name = name.lower()
        if name == 'relu':        return nn.ReLU()
        if name == 'sigmoid':     return nn.Sigmoid()
        if name == 'tanh':        return nn.Tanh()
        if name == 'softmax':     return nn.Softmax(dim=1)
        if name == 'leaky_relu':  return nn.LeakyReLU()

        raise NotImplementedError(f"Activation '{name}' is not supported.")

    @beartype
    @staticmethod
    def _safe_use_bias(layer_cfg: Dict[str, Any]) -> bool:

        val = layer_cfg.get('use_bias', True)
        return True if val is None else bool(val)

    @beartype
    def _build_model_modules(self) -> list[nn.Module]:

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

    @beartype
    def _build_criterion(self) -> Tuple:

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

    @beartype
    def _build_optimizer(self) -> optim.Optimizer:

        params = self.model.parameters()
        if self.opt_name == 'adam':
            return optim.Adam(params, lr=1e-3)
        if self.opt_name == 'sgd':
            # nesterov=True matches keras.optimizers.SGD(nesterov=True)
            return optim.SGD(params, lr=1e-2, momentum=0.9, nesterov=True)
        raise NotImplementedError(f"Optimizer '{self.opt_name}' not supported.")

    @beartype
    def _compileModel(self) -> None:
        self._build_criterion()   # raises on bad config

    @beartype
    def _createCheckpoint(self) -> None:
        os.makedirs(self.dirname, exist_ok=True)

    @beartype
    def _createModel(self) -> None:
        self.model = nn.Sequential(*self._build_model_modules())

    @beartype
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

    @beartype
    def loadBestModel(self) -> nn.Module:

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

    @beartype
    def quickTrain(self, decay_func: Optional[Callable[[int], float]] = None) -> nn.Module:

        self.best_val_loss = float('inf')
        self.decay_func    = decay_func

        self._createModel()
        self._compileModel()
        self._createCheckpoint()
        self._trainModel()

        return self.loadBestModel()

