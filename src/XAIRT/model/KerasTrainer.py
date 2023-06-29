from abc import ABC, ABCMeta, abstractmethod
from XAIRT.backend.types import OptionalList, OptionalSequence
from XAIRT.backend.types import VectorNumpy, MatrixNumpy, TensorNumpy
from XAIRT.backend.types import Dataset, Tensor

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle

__all__ = ["KerasTrainer", "TrainFullyConnected"]

class KerasTrainer(metaclass=ABCMeta):

	@abstractmethod
	def __init__(self):
		pass

	@abstractmethod
	def createModel(self):
		pass

	@abstractmethod
	def compileModel(self):
		pass

	@abstractmethod
	def checkpoint(self):
		pass

	@abstractmethod
	def trainModel(self):
		pass

	@abstractmethod
	def loadBestModel(self):
		pass

class TrainFullyConnected(KerasTrainer):

	def __init__(self, 
		     x: TensorNumpy | Dataset, y: TensorNumpy | Dataset, 
		     layers: list[list[int]], 
	             activation_layers: list[list[str]],
                     loss: OptionalList[str], 
		     optim: OptionalList[str], 
                     metrics: OptionalList[str],
                     batch_size: int, 
                     epochs: int, 
                     validation_split: float,
                     filename: str,
		     dirname: str):

		super().__init__()

		self.x = x
		self.y = y
		
		self.layers = layers
		self.activation_layers = activation_layers
		
		self.loss = loss
		self.optim = optim
		self.metrics = metrics
		
		self.batch_size = batch_size
		self.epochs = epochs
		self.validation_split = validation_split
		
		self.dirname = dirname
		self.filename = filename	

	def createModel(self):

		keras.backend.clear_session()
		
		inputs = Input(shape=(self.layers[0],))
		
		dense = Dense(self.layers[1], activation=self.activation_layers[0])		
		x = dense(inputs)
		for i in range(2, len(self.layers)):
		
			dense = Dense(self.layers[i], activation=self.activation_layers[i-1])
			x = dense(x) 
		
		self.model = Model(inputs=inputs, outputs=x)

	def compileModel(self):

		self.model.compile(loss=self.loss, optimizer=self.optim, metrics=self.metrics)

	def checkpoint(self):

		self.mod_h5 = os.path.join(self.dirname,
		                           self.filename + '.h5')
		self.mod_txt  = os.path.join(self.dirname,
		                             self.filename + '.txt')
		self.checkpoint = ModelCheckpoint(self.mod_h5, monitor='val_loss',
		                     verbose=1,save_best_only=True)
		
		self.callbacks = [self.checkpoint]
		

	def trainModel(self):
		self.fit = self.model.fit(self.x, self.y,
            			batch_size=self.batch_size,
            			epochs=self.epochs, 
            			shuffle=True,
            			validation_split = self.validation_split, 
            			callbacks=self.callbacks) 

	def loadBestModel(self):
		self.best_model = keras.models.load_model(self.mod_h5)

	def testOnePoint(self):

		pass


