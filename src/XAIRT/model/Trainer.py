from abc import ABCMeta, abstractmethod
from XAIRT.backend.types import Optional, OptionalList, TensorNumpy
from XAIRT.backend.types import Dataset, LayerDict, LossDict
from XAIRT.backend.types import Callable, Optimizer

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras import initializers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.regularizers import l2

from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression

import os

__all__ = ["Trainer", "TrainerNN", "TrainLR", "TrainFullyConnectedNN"]

class Trainer(metaclass=ABCMeta):

	@abstractmethod
	def __init__(self) -> None:
		pass

	@abstractmethod
	def _createModel(self) -> None:
		pass

	@abstractmethod
	def _trainModel(self) -> None:
		pass

	@abstractmethod
	def quickTrain(self) -> LinearRegression | Model:
		pass

class TrainerNN(Trainer):

	@abstractmethod
	def __init__(self) -> None:
		pass

	@abstractmethod
	def _compileModel(self) -> None:
		pass

	@abstractmethod
	def _createCheckpoint(self) -> None:
		pass

	@abstractmethod
	def loadBestModel(self) -> Model:
		pass

class TrainLR(Trainer):

	"""
	In an XAI context, only a normalized x and y make sense for XLR.
	Since all inputs should be of a similar scale to compare coeffs.
	"""

	def __init__(self, 
		    x: TensorNumpy, 
        	y: TensorNumpy,
            fit_intercept: Optional[bool] = False,
		    y_ref: Optional[float] = 0.0) -> None:

		super().__init__()

		self.x = x
		self.y = y
		self.fit_intercept = fit_intercept
		self.y_ref = y_ref
		self._model_state = []
		
	def _createModel(self) -> None:

		self.regr = LinearRegression(fit_intercept = self.fit_intercept)
		self._model_state.append('created')

	def _trainModel(self) -> None:

		self.regr.fit(self.x, self.y-self.y_ref)
		self._model_state.append('trained')

	def quickTrain(self) -> LinearRegression:

		self._model_state = []

		self._createModel()
		self._trainModel()

		return self.regr


class TrainFullyConnectedNN(TrainerNN):

	def __init__(self, 
		    	x: TensorNumpy | Dataset, y: TensorNumpy | Dataset, 
		    	layers: list[LayerDict],
            	losses: OptionalList[LossDict],
				optimizer: Optimizer,
                metrics: list[str],
                batch_size: int, 
                epochs: int, 
                validation_split: float,
                filename: str,
		     	dirname: str,
				random_nn_seed: Optional[int] = None,
				decay_rate: Optional[float] = None
				) -> None:

		super().__init__()

		self.x = x
		self.y = y
		
		self.layers = layers
		
		self.losses = losses
		self.optimizer = optimizer
		self.decay_rate = decay_rate
		self.metrics = metrics
		
		self.batch_size = batch_size
		self.epochs = epochs
		self.validation_split = validation_split
		
		self.dirname = dirname
		self.filename = filename	

		self.random_nn_seed = random_nn_seed

		self.model_metadata = {'layers' : self.layers,
				       		   'losses' : self.losses,
				       		   'optimizer'  : self.optimizer,
							   'decay_rate': self.decay_rate,
				       		   'metrics': self.metrics}
		self.train_metadata = {'batch_size'      : self.batch_size,
				       		   'epochs'          : self.epochs,
				       		   'validation_split': self.validation_split,
				       		   'filename'        : self.filename,
				       		   'dirname'         : self.dirname}
		self._model_state = []
		self.callbacks = []

	def _createModel(self) -> None:

		keras.backend.clear_session()

		_sizes = [layer['size'] for layer in self.layers]
		_activations = [layer['activation'] for layer in self.layers]
		_use_biases = [layer['use_bias'] if 'use_bias' in layer else None for layer in self.layers]
		_l2_w_regs = [layer['l2_w_reg'] if 'l2_w_reg' in layer else None for layer in self.layers]
		_l2_b_regs = [layer['l2_b_reg'] if 'l2_b_reg' in layer else None for layer in self.layers]

		if _activations[0] is not None:
			raise ValueError("Input layer cannot have an activation")
		if _use_biases[0] is not None:
			raise ValueError("Input layer cannot have a bias.")

		inputs = Input(shape=(_sizes[0],))
		dense = Dense(_sizes[1], 
					  activation=_activations[1], use_bias = _use_biases[1],
					  kernel_initializer=initializers.GlorotUniform(seed=self.random_nn_seed),
					  bias_initializer=initializers.Zeros(),
					  kernel_regularizer=l2(_l2_w_regs[1]),
					  bias_regularizer=l2(_l2_b_regs[1]))
		x = dense(inputs)
 
		for i in range(2, len(_sizes)):
		
			dense = Dense(_sizes[i], 
						  activation=_activations[i], use_bias = _use_biases[i],
					  	  kernel_initializer=initializers.GlorotUniform(seed=self.random_nn_seed),
					  	  bias_initializer=initializers.Zeros(),
						  kernel_regularizer=l2(_l2_w_regs[i]),
					  	  bias_regularizer=l2(_l2_b_regs[i]))
			x = dense(x)
		
		self.model = Model(inputs=inputs, outputs=x)

		self._model_state.append('created')

	def _compileModel(self) -> None:

		if len(self.losses) != 1:
			raise NotImplementedError("Weighted losses not implemented yet.")
		elif len(self.losses) == 1 and self.losses[0]['weight'] != 1.0:
			raise ValueError("Loss weight has to be 1.0 for a single loss function.")
		else: 
			pass

		_loss_kinds = [loss['kind'] for loss in self.losses]
		_loss_weights = [loss['weight'] for loss in self.losses]

		self.model.compile(loss=_loss_kinds[0], optimizer=self.optimizer, 
						   metrics=self.metrics)

		self._model_state.append('compiled')

	def _lrateSchedule(self, decay_func: Callable) -> None:

		self.lrate = LearningRateScheduler(decay_func)
		self.callbacks.append(self.lrate)

		self._model_state.append('lrate schedule')

	def _createCheckpoint(self) -> None:

		self.mod_h5 = os.path.join(self.dirname,
		                           self.filename + '.h5')
		self.mod_txt  = os.path.join(self.dirname,
		                             self.filename + '.txt')
		self.checkpoint = ModelCheckpoint(self.mod_h5, monitor='val_loss',
		                     			  verbose=1,save_best_only=True)
		
		self.callbacks.append(self.checkpoint)
		self._model_state.append('checkpointed')

	def _trainModel(self) -> None:

		self._fit = self.model.fit(self.x, self.y,
            			batch_size=self.batch_size,
            			epochs=self.epochs, 
            			shuffle=True,
            			validation_split = self.validation_split, 
            			callbacks=self.callbacks,
						verbose = 0) 

		self._model_state.append('trained')

	def loadBestModel(self) -> Model:

		if self._model_state[-1] != 'trained':
			raise Exception("Model is not trained!")

		best_model = keras.models.load_model(self.mod_h5)
	
		return best_model

	def quickTrain(self, decay_func: Optional[Callable] = None) -> Model:

		self._model_state = []

		self._createModel()
		self._compileModel()

		if decay_func is not None:
			self._lrateSchedule(decay_func)

		self._createCheckpoint()
		self._trainModel()

		return self.loadBestModel()


