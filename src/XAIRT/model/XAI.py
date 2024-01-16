from abc import ABCMeta, abstractmethod

from XAIRT.backend.types import Optional, OptionalList, Dict
from XAIRT.backend.types import TensorNumpy, LinearRegression
from XAIRT.backend.types import AnalysisNormalizeDict, AnalysisStatsDict
from XAIRT.backend.types import ModelMetadata, TrainMetadata
from XAIRT.backend.types import XAIMethodsDict

from XAIRT.backend.graph import getLayerIndexByName 

import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

import innvestigate
import innvestigate.utils as iutils
from innvestigate.analyzer.base import AnalyzerBase

import numpy as np

import warnings
import sys
import os
import pathlib

__all__ = ["X", "XAI", "XLR", "XAIR"]

class X(metaclass=ABCMeta):

	@abstractmethod
	def __init__(self) -> None:
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

class XAI(X):

	@abstractmethod
	def __init__(self) -> None:
		pass

	@abstractmethod
	def _create_analyzer(self) -> None:
		pass

class XLR(X):

	"""
	In an XAI context, only normalized samples makes sense for XLR
	Since all inputs should be of a similar scale to compare coeffs.
	"""

	def __init__(self,
		     	 model: OptionalList[LinearRegression],
		     	 samples: Optional[TensorNumpy] = None,
		     	 normalize: Optional[AnalysisNormalizeDict] = {'bool_':True, 'kind': 'Sum'}
		    	 ) -> None: 

		super().__init__()
		self.model = model
		self.samples = samples
		self.normalize = normalize
		self._coef = self.model.coef_
		self.fit_intercept = self.model.fit_intercept

	def _get_root(self,
		      	  model: OptionalList[LinearRegression]):

		if self.fit_intercept is False:

			self.x_tilde = np.zeros(self.samples.shape[1])

		else:

			raise NotImplementedError("fit_intercept has to be 0 for now, since it is unclear which root to choose for adjustment.")

	def _analyze_sample(self,
			    sample: TensorNumpy,
			    normalize: Optional[AnalysisNormalizeDict] = {'bool_':True, 'kind': 'Sum'}
			   ) -> TensorNumpy:

		a = self._coef * (sample - self.x_tilde)

		if normalize is None:

			normalize = {'bool_':False}

		elif normalize['bool_'] is True and 'kind' not in normalize:

			normalize['kind'] = 'Sum'

		else:

			pass

		if normalize['bool_'] is True and normalize['kind'] == 'MaxAbs':

			a /= np.max(np.abs(a))

		elif normalize['bool_'] is True and normalize['kind'] == 'Sum':

			a /= np.sum(a)

		elif normalize['bool_'] is True and normalize['kind'] != 'MaxAbs' and normalize['kind'] != 'Sum':

			raise NotImplementedError("Only MaxAbs and Sum normalization currently available!")

		else:

			pass

		return a

	def analyze_samples(self,
			    		samples: TensorNumpy,
			    		normalize: Optional[AnalysisNormalizeDict] = {'bool_':True, 'kind': 'Sum'}
			    		) -> TensorNumpy:

		a = np.zeros(samples.shape, dtype = np.float64)

		numSamples = samples.shape[0]

		for i in range(numSamples):

			a[i] = self._analyze_sample(samples[i], normalize)

		return a

	def quick_analyze(self) -> tuple[TensorNumpy, AnalysisStatsDict]:

		if self.model is not None and self.samples is not None:
		
			self._get_root(self.model)

			if self.normalize is None:
				self.normalize = {'bool_': False, 'kind': ''}

			a = self.analyze_samples(self.samples, self.normalize)

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

class XAIR(XAI):

	def __init__(self, 
		     	 model: Optional[Model],
		     	 method: Optional[XAIMethodsDict] = None,
		     	 kind: Optional[str] = None,
		     	 samples: Optional[TensorNumpy] = None,
		     	 normalize: Optional[AnalysisNormalizeDict] = {'bool_':True, 'kind': 'Sum'},
		     	 #**kwargs: Unpack[LetzgusDict], #Will be compatible with Python 3.12
		     	 **kwargs: Dict
		    	 ) -> None:

		super().__init__()
		self.model = model
		self.method = method
		self.kind = kind
		self.samples = samples
		self.normalize = normalize
		self.y_ref = kwargs['y_ref'] if kwargs.__contains__('y_ref') else 0.0
		self.kwargs = kwargs
		self.models_letzgus = []

	def _create_analyzer(self, 
                         method: XAIMethodsDict, 
                         kind: str,
			     		 sample: Optional[TensorNumpy] = None,
			     		 #**kwargs: Unpack[LetzgusDict], #Will be compatible with Python 3.12 
		             	 **kwargs: Dict) -> OptionalList[AnalyzerBase]:
		
		if kind == 'classic':

			Analyze = innvestigate.create_analyzer(method['name'], self.model, **method['optParams'])

		elif kind == 'letzgus':

			if bool(kwargs) is False:

				raise ValueError("No Letzgus hyperparameters given!")

			if sample is None:

				raise ValueError("No sample to create letzgus analyzer!")			

			models_letzgus = self.createLetzgus(sample, **kwargs)

			Analyze = [innvestigate.create_analyzer(method['name'], self.model, **method['optParams']),
				   innvestigate.create_analyzer(method['name'], self.model, **method['optParams']),
				   innvestigate.create_analyzer(method['name'], self.model, **method['optParams'])]

		else:

			raise NotImplementedError("The only kinds of analyzers available are classic and letzgus!")

		return Analyze

	def _analyze_sample(self,
			    		method: XAIMethodsDict,
			    		kind: str,
			    		sample: Optional[TensorNumpy],
			    		normalize: Optional[AnalysisNormalizeDict] = {'bool_':True, 'kind': 'Sum'},
			    		Analyze: OptionalList[AnalyzerBase] = None,
			    		#**kwargs: Unpack[LetzgusDict], #Will be compatible with Python 3.12
			    		**kwargs: Dict
			  			) -> TensorNumpy:
		
			
		a = np.zeros(sample.shape, dtype = np.float64)

		if kind =='classic' and Analyze is not None:

			if sample is None:

				raise ValueError("No sample for classic analyzer to analyze!")
	
			if Analyze is list:
				raise ValueError("classic analyzer cannot be a list!")

			a = Analyze.analyze(sample[np.newaxis,:])

		elif kind == 'letzgus' and Analyze is not None:

			if bool(kwargs) is False:

				raise ValueError("No Letzgus hyperparameters given!")

			if sample is None:

				raise ValueError("No sample to create letzgus analyzer!")			

			if sample is not None and kwargs.__contains__('sampleLetzgus') is False:

				warnings.warn("Letzgus might be analyzing a different sample than intended.")

			elif sample is not None and kwargs.__contains__('sampleLetzgus'):

				warnings.warn("Letzgus might be analyzing a different sample than intended.")
				print(f"Max delta samples = {np.max(np.abs(sample-kwargs['sample']))}")

			elif sample is None and kwargs.__contains__('sampleLetzgus'):

				sample = kwargs['sampleLetzgus']

			else:

				raise ValueError("No sample for letzgus analyzer to be created!")

			if Analyze is not list:

				raise TypeError("letzgus Analyzer has to be a list of 3 analyzers!")

			if len(Analyze) != 3:

				raise ValueError("letzgus Analyzer has to be a list of 3 analyzers!")

			a = Analyze[0].analyze(sample[np.newaxis,:]) \
              + Analyze[1].analyze(sample[np.newaxis,:]) \
              + Analyze[2].analyze(sample[np.newaxis,:])

		elif kind == 'classic' and Analyze is None:

			if sample is None:

				raise ValueError("No sample for classic analyzer to analyze!")
			

			Analyze = self._create_analyzer(method, kind, sample, **kwargs)

			a = Analyze.analyze(sample[np.newaxis,:])

		elif kind == 'letzgus' and Analyze is None:

			if bool(kwargs) is False:

				raise ValueError("No Letzgus hyperparameters given!")

			if sample is None:

				raise ValueError("No sample to create letzgus analyzer!")			

			if sample is not None and kwargs.__contains__('sampleLetzgus') is False:

				warnings.warn("Letzgus might be analyzing a different sample than intended.")

			elif sample is not None and kwargs.__contains__('sampleLetzgus'):

				warnings.warn("Letzgus might be analyzing a different sample than intended.")
				print(f"Max delta samples = {np.max(np.abs(sample-kwargs['sample']))}")

			elif sample is None and kwargs.__contains__('sampleLetzgus'):

				sample = kwargs['sampleLetzgus']

			else:

				raise ValueError("No sample for letzgus analyzer to be created!")

			Analyze = self._create_analyzer(method, kind, sample, **kwargs)

			a = Analyze[0].analyze(sample[np.newaxis,:]) \
              + Analyze[1].analyze(sample[np.newaxis,:]) \
              + Analyze[2].analyze(sample[np.newaxis,:])

		else:

			raise NotImplementedError("The only kinds of analyzers available are classic and letzgus!")

		if normalize is None:

			normalize = {'bool_':False}

		elif normalize['bool_'] is True and 'kind' not in normalize:

			normalize['kind'] = 'Sum'

		else:

			pass

		if normalize['bool_'] is True and normalize['kind'] == 'MaxAbs':

			a /= np.max(np.abs(a))

		elif normalize['bool_'] is True and normalize['kind'] == 'Sum':

			a /= np.sum(a)

		elif normalize['bool_'] is True and normalize['kind'] != 'MaxAbs' and normalize['kind'] != 'Sum':

			raise NotImplementedError("Only MaxAbs and Sum normalization currently available!")

		else:

			pass

		return a
		
	def analyze_samples(self,
			    		method: XAIMethodsDict,
			    		kind: str,
			    		samples: TensorNumpy,
                        normalize: AnalysisNormalizeDict = {'bool_':True, 'kind': 'Sum'},
			    		Analyze: OptionalList[AnalyzerBase] = None,
			    		**kwargs: Dict
                    	) -> TensorNumpy:

		a = np.zeros(samples.shape, dtype = np.float64)
		numSamples = samples.shape[0]

		for i in range(numSamples):

			a[i] = self._analyze_sample(method, kind, samples[i], normalize, Analyze, **kwargs)

		return a

	def offsetLetzgus(self, 
			  sample: TensorNumpy, 
			  y_ref: float, 
			  step_width: float = 0.00005, 
			  max_it: int = 10e4, 
			  method_reg: str = "flooding") -> TensorNumpy:
		
		### Finding _a_ref for a given y_ref

		if method_reg == "flooding":

			_model_part = Model(inputs=self.model.input,
                          		   outputs=self.model.layers[-2].output)
			_a_ref = _model_part.predict(sample[np.newaxis,:])[0,:]
			_a_ref = _a_ref[:, np.newaxis]
			_update = np.ones(_a_ref.shape) * step_width
			_y = self.model.predict(sample[np.newaxis,:])

			_counter = 0

			if _y >= y_ref:

				while _y >= y_ref:				

					_a_ref = np.maximum(np.zeros(_a_ref.shape),_a_ref-_update)
					_y = np.dot(self.model.layers[-1].get_weights()[0][:,0], _a_ref[:,0])
					_counter +=1 
					print(f'iteration {_counter} - y: {_y}', end='\r')	
					if _counter > max_it:
						print(f'! reference value {y_ref} was not reached within {round(max_it)} iterations!')
						break

			else:	

				while _y <= y_ref:		

					_a_ref = np.maximum(np.zeros(_a_ref.shape),_a_ref+_update)
					_y = np.dot(self.model.layers[-1].get_weights()[0][:,0], _a_ref[:,0])
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
		if len(self.model.layers[-1].get_weights()) > 1:
			bias_out = self.model.layers[-1].get_weights()[1]

		savepath = self.model.save('model_orig.h5')

		model1 = keras.models.load_model(str(pathlib.Path().resolve()) + '/model_orig.h5')
		model2 = keras.models.load_model(str(pathlib.Path().resolve()) + '/model_orig.h5')
		model3 = keras.models.load_model(str(pathlib.Path().resolve()) + '/model_orig.h5')

		model1.layers[-2].set_weights([W_in, bias_in-_a_ref[:,0]])
		model2.layers[-2].set_weights([-W_in, -bias_in])
		model3.layers[-2].set_weights([-W_in, -bias_in+_a_ref[:,0]])

		if len(self.model.layers[-1].get_weights()) > 1:

			model1.layers[-1].set_weights([ W_out, bias_out])
			model2.layers[-1].set_weights([ W_out, bias_out])
			model3.layers[-1].set_weights([-W_out, bias_out])

		else:

			model1.layers[-1].set_weights([W_out])
			model2.layers[-1].set_weights([W_out])
			model3.layers[-1].set_weights([-W_out])

		return [model1, model2, model3]

	def createLetzgus(self,
			     sample: TensorNumpy, 
			     y_ref: float, 
			     step_width: float = 0.00005, 
			     max_it: int = 10e4, 
			     method_reg: str = "flooding") -> tuple[Model, Model, Model]:

		_a_ref = self.offsetLetzgus(sample, y_ref, step_width, max_it, method_reg)
		return self.triplicateLetzgus(_a_ref)

	def quick_analyze(self) -> tuple[TensorNumpy, AnalysisStatsDict]:

		if self.model is not None and self.method is not None and self.kind is not None and self.samples is not None and bool(self.kwargs):
			
			if self.normalize is None:
				self.normalize = {'bool_': False, 'kind': ''}

			if self.kind == 'letzgus':
				a = self.analyze_samples(self.method, self.kind, self.samples, self.normalize, **self.kwargs)
			else:
				Analyze = self._create_analyzer(self.method, self.kind, **self.kwargs)
				a = self.analyze_samples(self.method, self.kind, self.samples, self.normalize, Analyze, **self.kwargs)

			statistics = self.compute_statistics(a)

			return a, statistics

		else:
			raise ValueError("Not enough information provided to analyze automatically!")

	def check_sample(self, sample):

		y = self.model.predict(sample[np.newaxis, :])
		model1, model2, model3 = self.createLetzgus(sample, self.y_ref)
		y_reg = model1.predict(sample[np.newaxis, :]) + model2.predict(sample[np.newaxis, :]) + model3.predict(sample[np.newaxis, :])

		return y - y_reg - self.y_ref

	@staticmethod
	def compute_statistics(a: TensorNumpy) -> AnalysisStatsDict:
		
		Stats = {}

		### Mean heatmap over all samples
		Stats['mean'] = np.mean(a, axis = 0)
		### TODO - Add more stats

		return Stats
