from abc import ABCMeta, abstractmethod
from XAIRT.backend.types import OptionalList, TensorNumpy
from XAIRT.backend.types import Dataset, LayerDict, LossDict

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint

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

	def __init__(self, 
		     x: TensorNumpy, 
                     y: TensorNumpy
                     fit_intercept: bool = False) -> None:

		super().__init__()

		self.x = x
		self.y = y
                self.fit_intercept = fit_intercept
                self._model_state = []
		
	def _createModel(self) -> None:

                self.regr = LinearRegression(fit_intercept = self.fit_intercept)
                self._model_state.append('created')

	def _trainModel(self) -> None:

                self.regr.fit(x, y)
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
		     optim: str, 
                     metrics: list[str],
                     batch_size: int, 
                     epochs: int, 
                     validation_split: float,
                     filename: str,
		     dirname: str) -> None:

		super().__init__()

		self.x = x
		self.y = y
		
		self.layers = layers
		
		self.losses = losses
		self.optim = optim
		self.metrics = metrics
		
		self.batch_size = batch_size
		self.epochs = epochs
		self.validation_split = validation_split
		
		self.dirname = dirname
		self.filename = filename	

		self.model_metadata = {'layers' : self.layers,
				       'losses' : self.losses,
				       'optim'  : self.optim,
				       'metrics': self.metrics}
		self.train_metadata = {'batch_size'      : self.batch_size,
				       'epochs'          : self.epochs,
				       'validation_split': self.validation_split,
				       'filename'        : self.filename,
				       'dirname'         : self.dirname}
		self._model_state = []

	def _createModel(self) -> None:

		keras.backend.clear_session()

		_sizes = [layer['size'] for layer in self.layers]
		_activations = [layer['activation'] for layer in self.layers]
		_use_biases = [layer['use_bias'] if 'use_bias' in layer else None for layer in self.layers]

		if _activations[0] is not None:
			raise ValueError("Input layer cannot have an activation")
		if _use_biases[0] is not None:
			raise ValueError("Input layer cannot have a bias.")

		inputs = Input(shape=(_sizes[0],))
		dense = Dense(_sizes[1], activation=_activations[1], use_bias = _use_biases[1])
		x = dense(inputs)
 
		for i in range(2, len(_sizes)):
		
			dense = Dense(_sizes[i], activation=_activations[i], use_bias = _use_biases[i])
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

		self.model.compile(loss=_loss_kinds[0], optimizer=self.optim, metrics=self.metrics)

		self._model_state.append('compiled')

	def _createCheckpoint(self) -> None:

		self.mod_h5 = os.path.join(self.dirname,
		                           self.filename + '.h5')
		self.mod_txt  = os.path.join(self.dirname,
		                             self.filename + '.txt')
		self.checkpoint = ModelCheckpoint(self.mod_h5, monitor='val_loss',
		                     verbose=1,save_best_only=True)
		
		self.callbacks = [self.checkpoint]
		
		self._model_state.append('checkpointed')

	def _trainModel(self) -> None:

		self._fit = self.model.fit(self.x, self.y,
            			batch_size=self.batch_size,
            			epochs=self.epochs, 
            			shuffle=True,
            			validation_split = self.validation_split, 
            			callbacks=self.callbacks) 

		self._model_state.append('trained')

	def loadBestModel(self) -> Model:

		if self._model_state[-1] != 'trained':
			raise Exception("Model is not trained!")

		best_model = keras.models.load_model(self.mod_h5)
	
		return best_model

	def quickTrain(self) -> Model:

		self._model_state = []

		self._createModel()
		self._compileModel()
		self._createCheckpoint()
		self._trainModel()

		return self.loadBestModel()


