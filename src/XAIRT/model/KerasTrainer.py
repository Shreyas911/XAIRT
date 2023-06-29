from abc import ABC, ABCMeta, abstractmethod

__all__ = ["KerasTrainer", "TrainFullyConnected"]

class KerasTrainer(metaclass=ABCMeta):

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

	@abstractmethod
	def testOnePoint(self):
		pass

class TrainFullyConnected(KerasTrainer):
	def __init__(self, a):
		super().__init__()
		self.a = a
	
	def createModel(self):
		pass

	def compileModel(self):
		pass

	def checkpoint(self):
		pass

	def trainModel(self):
		pass

	def loadBestModel(self):
		pass

	def testOnePoint(self):
		pass


