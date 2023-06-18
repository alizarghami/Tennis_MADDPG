
# Solving Tennis problem from Unity Environments using Deep Reinforcement Learning

### Introduction

This repository contains code for training intelligent agents that can act efficiently in Tennis unity environment using Deep Reinforcement Learning.

### About the environment

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The goal is to reach score of +0.5 over 100 consecutive episodes for a single agent.

### Getting Started

In order to run this code you need to have Python 3 and Jupyter-notebook installed. In addition you need to install the following modules.
* Pytorch: [click here](https://pytorch.org/get-started/locally)
* Numpy: [click here](https://numpy.org/install)
* UnityEnvironment: [click here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)
* OpenAI Gym: [click here](https://github.com/openai/gym)

You also need to download the Tennis environment from the links below. You need only select the environment that matches your operating system:
* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
Make sure to decompress the zipped file before running the code

You are strongly suggested to install all the dependencies in a virtual environment. If you are using conda you can create and activate a virtual environment by the following commands:

	conda create --name ENVIRONMENT_NAME python=3.6
	conda activate ENVIRONMENT_NAME
	
You can deactivate your environment by this command:

	conda deactivate
	
An alternative method for using python virtual environments can be found here: [click here](https://virtualenv.pypa.io/en/latest/)

For more information and instructions on how to install all dependencies check [this link](https://github.com/udacity/deep-reinforcement-learning#dependencies).

### Instructions

In order to run the code you need to open `tennis.ipynb` in your Jupyter-notebook. Point to the Navigation Environment location on your system where specified in the code and run.
