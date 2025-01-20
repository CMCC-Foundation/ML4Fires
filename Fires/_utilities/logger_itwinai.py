# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 					Copyright 2024 - CMCC Foundation						
#																			
# Site: 			https://www.cmcc.it										
# CMCC Institute:	IESP (Institute for Earth System Predictions)
# CMCC Division:	ASC (Advanced Scientific Computing)						
# Author:			Emanuele Donno											
# Email:			emanuele.donno@cmcc.it									
# 																			
# Licensed under the Apache License, Version 2.0 (the "License");			
# you may not use this file except in compliance with the License.			
# You may obtain a copy of the License at									
#																			
#				https://www.apache.org/licenses/LICENSE-2.0					
#																			
# Unless required by applicable law or agreed to in writing, software		
# distributed under the License is distributed on an "AS IS" BASIS,			
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.	
# See the License for the specific language governing permissions and		
# limitations under the License.											
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import os
import joblib
import lightning.pytorch as lp
from itwinai.loggers import Logger, Prov4MLLogger, ConsoleLogger, MLFlowLogger
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from Fires._utilities.decorators import export

@export
class SimpleItwinaiLogger(Logger):
	"""Simplified logger.

	Args:
		savedir (str, optional): where to store artifacts.
			Defaults to 'mllogs'.
		log_freq (Union[int, Literal['epoch', 'batch']], optional):
			determines whether the logger should fulfill or ignore
			calls to the `log()` method. See ``Logger.should_log`` method for
			more details. Defaults to 'epoch'.
		log_on_workers (Optional[Union[int, List[int]]]): if -1, log on all
			workers; if int log on worker with rank equal to log_on_workers;
			if List[int], log on workers which rank is in the list.
			Defaults to 0 (the global rank of the main worker).
	"""

	def __init__(
	 self,
	 savedir: str = 'mllogs',
	 log_freq: Union[int, Literal['epoch', 'batch']] = 'epoch',
	 log_on_workers: Union[int, List[int]] = 0
	) -> None:
		super().__init__(savedir=savedir, log_freq=log_freq, log_on_workers=log_on_workers)

		self.savedir = savedir
		os.makedirs(self.savedir, exist_ok=True)
		self._hyperparams = {}
		self.supported_kinds = ('torch', 'metric', 'artifact')
		self.worker_rank = None


	def create_logger_context(self, rank: Optional[int] = None) -> Any:
		"""
		Initializes the logger context.

		Args:
			rank (Optional[int]): global rank of current process,
				used in distributed environments. Defaults to None.
		"""
		self.worker_rank = rank

		if not self.should_log(): return
		os.makedirs(self.savedir, exist_ok=True)
		run_dirs = sorted([int(dir) for dir in os.listdir(self.savedir)])
		self.run_id = 0 if len(run_dirs) == 0 else int(run_dirs[-1]) + 1
		self.run_path = os.path.join(self.savedir, str(self.run_id))
		os.makedirs(self.run_path)


	def destroy_logger_context(self) -> None:
		"""Destroy logger. Do nothing."""
		if not self.should_log(): return


	def save_hyperparameters(self, params: Dict[str, Any]) -> None:
		"""Save hyperparameters. Do nothing.

		Args:
			params (Dict[str, Any]): hyperparameters dictionary.
		"""
		if not self.should_log():
			return

		hyperparams_path = os.path.join(self.savedir, "hyperparameters.joblib")
		joblib.dump(params, hyperparams_path)


	def log(
	 self,
	 item: Union[Any, List[Any]],
	 identifier: Union[str, List[str]],
	 kind: str = 'metric',
	 step: Optional[int] = None,
	 batch_idx: Optional[int] = None,
	 **kwargs
	) -> None:
		"""Print metrics to stdout and save artifacts to the filesystem.

		Args:
			item (Union[Any, List[Any]]): element to be logged (e.g., metric).
			identifier (Union[str, List[str]]): unique identifier for the
				element to log(e.g., name of a metric).
			kind (str, optional): type of the item to be logged. Must be
				one among the list of ``self.supported_kinds``.
				Defaults to 'metric'.
			step (Optional[int], optional): logging step. Defaults to None.
			batch_idx (Optional[int], optional): DataLoader batch counter
				(i.e., batch idx), if available. Defaults to None.
			kwargs: keyword arguments to pass to the logger.
		"""

		if not self.should_log(batch_idx=batch_idx):
			return

		if kind == 'artifact':
			if isinstance(item, str) and os.path.isfile(item):
				import shutil
				identifier = os.path.join(
				 self.run_path,
				 identifier
				)
				if len(os.path.dirname(identifier)) > 0:
					os.makedirs(os.path.dirname(identifier), exist_ok=True)
				print(f"Serializing to {identifier}...")
				shutil.copyfile(item, identifier)
			else:
				identifier = os.path.join(
				 os.path.basename(self.run_path),
				 identifier
				)
				print(f"Serializing to {identifier}...")
				self.serialize(item, identifier)
		elif kind == 'torch':
			identifier = os.path.join(self.run_path, identifier)
			print(f"Saving to {identifier}...")
			import torch
			torch.save(item, identifier)
		else:
			print(f"{identifier} = {item}")


@export
class ItwinaiLightningLogger(lp.loggers.Logger):
	def __init__(self, savedir: str, name: str = "itwinai", version:Optional[int | None] = None):
		super().__init__()
		self.logger = SimpleItwinaiLogger(savedir=savedir) # ConsoleLogger(savedir=savedir)
		self._name = name
		self._version = version if version is not None else 0
		self.logger.worker_rank = None

	@property
	def name(self) -> str:
		"""Nome del logger."""
		return self._name

	@property
	def version(self) -> str:
		"""Versione del logger."""
		return self._version

	@property
	def experiment(self) -> Any:
		"""Ritorna l'istanza dell'esperimento. Può essere utilizzato per accedere a funzionalità specifiche del logger."""
		return self.logger

	def log_hyperparams(self, params: Dict[str, Any]) -> None:
		"""Log dei parametri iper."""
		self.logger.save_hyperparameters(params)

	def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
		"""Log delle metriche."""
		for key, value in metrics.items():
			self.logger.log(
			 item=value,
			 identifier=key,
			 step=step
			)

	def save(self) -> None:
		"""Salva lo stato del logger, se necessario."""
		# Implementa la logica di salvataggio se necessaria
		pass

	def finalize(self, status: str) -> None:
		"""Finalizza il logger al termine dell'esperimento."""
		self.logger.destroy_logger_context()



@export
class ItwinaiMLFlowLogger(lp.loggers.Logger):
	def __init__(
			self,
			savedir: str, 
			name: str = "itwinai", 
			version:Optional[int | None] = None,
			experiment_name:str = "experiment_name",
			run_name:Optional[str | None] = None):
		super().__init__()
		self.logger = MLFlowLogger(
			savedir=savedir,
			experiment_name=experiment_name,
			run_name=run_name,
			tracking_uri=os.getenv('MLFLOW_TRACKING_URI')
		)
		self._name = name
		self._version = version if version is not None else 0
		self.logger.worker_rank = None

	@property
	def name(self) -> str:
		"""Nome del logger."""
		return self._name

	@property
	def version(self) -> str:
		"""Versione del logger."""
		return self._version

	@property
	def experiment(self) -> Any:
		"""Ritorna l'istanza dell'esperimento. Può essere utilizzato per accedere a funzionalità specifiche del logger."""
		return self.logger

	def log_hyperparams(self, params: Dict[str, Any]) -> None:
		"""Log dei parametri iper."""
		self.logger.save_hyperparameters(params)

	def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
		"""Log delle metriche."""
		for key, value in metrics.items():
			self.logger.log(
			 item=value,
			 identifier=key,
			 step=step
			)

	def save(self) -> None:
		"""Salva lo stato del logger, se necessario."""
		# Implementa la logica di salvataggio se necessaria
		pass

	def finalize(self, status: str) -> None:
		"""Finalizza il logger al termine dell'esperimento."""
		self.logger.destroy_logger_context()


@export
class ProvenanceLogger(lp.loggers.Logger):
	def __init__(self,
		savedir: str,
		name: str = "provenance",
		version:Optional[int | None] = None,
		prov_user_namespace="www.example.org",
		experiment_name="experiment_name",
		save_after_n_logs = 1,
		create_graph = False,
		create_svg = False,
		log_freq = 'epoch',
		log_on_workers = 0):
		super().__init__()

		os.makedirs(savedir, exist_ok=True)
		
		self.logger = Prov4MLLogger(
			prov_user_namespace = prov_user_namespace,
			experiment_name = experiment_name,
			provenance_save_dir = savedir,
			save_after_n_logs = save_after_n_logs,
			create_graph = create_graph,
			create_svg = create_svg,
			log_freq = log_freq,
			log_on_workers = log_on_workers,
		)
		self._name = name
		self._version = version if version is not None else 0
		self.logger.worker_rank = None

	@property
	def name(self) -> str:
		"""Nome del logger."""
		return self._name

	@property
	def version(self) -> str:
		"""Versione del logger."""
		return self._version

	@property
	def experiment(self) -> Any:
		"""Ritorna l'istanza dell'esperimento. Può essere utilizzato per accedere a funzionalità specifiche del logger."""
		return self.logger

	def log_hyperparams(self, params: Dict[str, Any]) -> None:
		"""Log degli iperparametri"""
		self.logger.save_hyperparameters(params)

	def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
		"""Log delle metriche."""
		for key, value in metrics.items():
			self.logger.log(
				item=value,
				identifier=key,
				step=step
			)

	def save(self) -> None:
		"""Salva lo stato del logger, se necessario."""
		# Implementa la logica di salvataggio se necessaria
		pass

	def finalize(self, status: str) -> None:
		"""Finalizza il logger al termine dell'esperimento."""
		self.logger.destroy_logger_context()