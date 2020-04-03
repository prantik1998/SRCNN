import click

from config.config import config

from src.pipelinemanager import pipelinemanager

@click.group()
def main():
	pass

@main.command()
def train():
	manager.train()

@main.command()
def test():
	manager.test()

@main.command()
def visualise():
	manager.visualise()


if __name__=="__main__":
	manager = pipelinemanager(config)
	main()