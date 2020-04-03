import torch

from .train import train
from .visualise import visualise
from .test import test


class pipelinemanager:
	def __init__(self,config):
		self.config = config

	def train(self):
		train(self.config)

	def test(self):
		test(self.config)

	def visualise(self):
		visualise(self.config)
		

