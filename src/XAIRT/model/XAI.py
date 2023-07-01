from abc import ABCMeta, abstractmethod

from XAIRT.backend.types import Optional, OptionalList
from XAIRT.backend.types import TensorNumpy
from XAIRT.backend.types import AnalysisNormalizeDict, AnalysisStatsDict

from tensorflow.keras.models import Model

import innvestigate
import innvestigate.utils as iutils
from innvestigate.analyzer.base import AnalyzerBase

import numpy as np

__all__ = ["XAI", "XAIR"]

class XAI(metaclass=ABCMeta):

	@abstractmethod
	def __init__(self) -> None:
		pass

	@abstractmethod
	def _create_analyzer(self) -> None:
		pass

	@abstractmethod
	def _analyze_sample(self) -> TensorNumpy:
		pass

	@abstractmethod
	def analyze_samples(self) -> TensorNumpy:
		pass

	@abstractmethod
	def quick_analyze(self) -> tuple[TensorNumpy, AnalysisStatsDict]:
		pass

	@abstractmethod
	def compute_statistics(self) -> AnalysisStatsDict:
		pass


class XAIR(XAI):

	def __init__(self, 
		     model: Model,
		     method: Optional[str] = None,
		     samples: Optional[TensorNumpy] = None,
		     normalize: Optional[AnalysisNormalizeDict] = {'bool_':True, 'kind': 'MaxAbs'},
		     y_ref: Optional[float] = 0.0
		    ) -> None:

		super().__init__()
		self.model = model
		self.method = method
		self.samples = samples
		self.normalize = normalize
		self.y_ref = y_ref

	def _create_analyzer(self, method: str) -> AnalyzerBase:
		
		Analyze = innvestigate.create_analyzer(method, self.model) 
		return Analyze

	def _analyze_sample(self,
			    method: str, 
			    sample: TensorNumpy,
			    normalize: Optional[AnalysisNormalizeDict] = {'bool_':True, 'kind': 'MaxAbs'},
			    Analyze: Optional[AnalyzerBase] = None
			  ) -> TensorNumpy:
		
		if Analyze is None:
			Analyze = self._create_analyzer(method)
		if isinstance(Analyze, AnalyzerBase) is False:
			raise ValueError("Analyzer has to be an instance of some subclass of AnalyzerBase!")
		a = np.zeros(sample.shape, dtype = np.float64)
		a = Analyze.analyze(sample[np.newaxis,:])
		
		if normalize is None:
			normalize = {'bool_':False}
		elif normalize['bool_'] is True and 'kind' not in normalize:
			normalize['kind'] = 'MaxAbs'
		else:
			pass

		if normalize['bool_'] is True and normalize['kind'] == 'MaxAbs':
			a /= np.max(np.abs(a))
		elif normalize['bool_'] is True and normalize['kind'] != 'MaxAbs':
			raise NotImplementedError("Only MaxAbs normalization currently available!")
		else:
			pass

		return a
		
	def analyze_samples(self,
			    method: str,
			    samples: TensorNumpy,
                            normalize: AnalysisNormalizeDict = {'bool_':True, 'kind': 'MaxAbs'}
                          ) -> TensorNumpy:

		Analyze = self._create_analyzer(method)
		a = np.zeros(samples.shape, dtype = np.float64)

		### TODO - Don't normalize here, do it in a vectorized fashion after

		numSamples = samples.shape[0]

		for i in range(numSamples):
			a[i] = self._analyze_sample(method, samples[i], normalize, Analyze)

		### Commented due to bug
		#if normalize is None:
		#	normalize = {'bool_':False}
		#elif normalize['bool_'] is True and 'kind' not in normalize:
		#	normalize['kind'] = 'MaxAbs'
		#else:
		#	pass
		#
		#if normalize['bool_'] is True and normalize['kind'] == 'MaxAbs':
		#	### Axes to normalize over
		#	axes = tuple(np.arange(np.ndim(a))[1:])
		#	maxAbsSample = np.max(np.abs(a), axis = axes)
		#	
		#	"""
		#	np.less_equal.outer(maxAbsSample, a), makes number of dimensions of
		#	maxAbsSample == number of dimensions of a by adding many np.newaxis
		#	"""
		#	print(
		#	a = np.divide(a, np.less_equal.outer(maxAbsSample, a))
		#elif normalize['bool_'] is True and normalize['kind'] != 'MaxAbs':
		#	raise NotImplementedError("Only MaxAbs normalization currently available!")
		#else:
		#	pass

		return a

	def offsetLetzgus(self, 
			  sample: TensorNumpy, 
			  y_ref: float, 
			  step_width: float = 0.00005, 
			  max_it: int = 10e4, 
			  method_reg: str = "flooding") -> TensorNumpy:
		
		### Finding _a_ref for a given y_ref

		if method_reg == "flood":

			_model_part = Model(inputs=self.model.input,
                          		   outputs=self.model.layers[-2].output)
			_a_ref = _model_part.predict(sample[np.newaxis,:])[0,:]
			_a_ref = _a_ref[:, np.newaxis]
			_update = np.ones(_a_ref.shape) * step_width
			_y = self.model.predict(sample[np.newaxis,:])

			_counter = 0

			if _y >= y_ref:

				_a_ref = np.maximum(np.zeros(_a_ref.shape),_a_ref-update)
				_y = np.dot(model.layers[-1].get_weights()[0][:,0], _a_ref[:,0])
				_counter +=1 
				print(f'iteration {_counter} - y: {_y}', end='\r')	
				if _counter > max_it:
					print(f'! reference value {y_ref} was not reached within {round(max_it)} iterations!')
					break

			else:			

				_a_ref = np.maximum(np.zeros(_a_ref.shape),_a_ref+update)
				_y = np.dot(model.layers[-1].get_weights()[0][:,0], _a_ref[:,0])
				_counter +=1 
				print(f'iteration {_counter} - y: {_y}', end='\r')	
				if _counter > max_it:
					print(f'! reference value {y_ref} was not reached within {round(max_it)} iterations!')
					break

		else:

			raise NotImplementedError("The only method_reg available are : flooding")

		return _a_ref

	def triplicateLetzgus(self,
			      _a_ref: TensorNumpy) -> tuple[Model, Model, Model]:

		# get weights and biases
		W_in = self.model.layers[-2].get_weights()[0]
		W_out = self.model.layers[-1].get_weights()[0]
		bias_in = self.model.layers[-2].get_weights()[1]
		
		return model1, model2, model3

	def benchmarkLetzgus(self,
			     sample: TensorNumpy, 
			     y_ref: float, 
			     step_width: float = 0.00005, 
			     max_it: int = 10e4, 
			     method_reg: str = "flooding") -> tuple[Model, Model, Model]:

		_a_ref = self.offsetLetzgus(sample, y_ref, step_width, max_it, method_reg)
		model1, model2, model3 = self.triplicateLetzgus(_a_ref)

		return model1, model2, model3

	def quick_analyze(self) -> tuple[TensorNumpy, AnalysisStatsDict]:

		if self.model is not None and self.method is not None and self.samples is not None:
			if self.normalize is None:
				self.normalize = {'bool_':False, 'kind': ''}
			A = self._create_analyzer(self.method)
			a = self.analyze_samples(self.method, self.samples, self.normalize)
			statistics = self.compute_statistics(a)
			return a, statistics

		else:
			raise ValueError("Not enough information provided to analyze automatically!")


		

	@staticmethod
	def compute_statistics(a: TensorNumpy) -> AnalysisStatsDict:
		
		Stats = {}

		### Mean heatmap over all samples
		Stats['mean'] = np.mean(a, axis = 0)
		### TODO - Add more stats

		return Stats	
