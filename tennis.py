# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 12:51:20 2023

@author: Ali
"""

from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from maddpg_agent import Agent


class Tennis(object):
    def __init__(self, env_path, criteria=0.5, seed=0):
        """
        Creates a Navigtion instance

        Parameters
        ----------
        env_path : str
            Path to the unity environment.
        criteria : int, optional
            The score we aim to reach. The default is 0.5.
        seed : int, optional
            Seed to use. The default is 0.
        """
        self.env = UnityEnvironment(file_name = env_path)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]   
        self.env_info = self.env.reset(train_mode=True)[self.brain_name]
        
        self.action_size = self.brain.vector_action_space_size
        self.state_size = self.env_info.vector_observations.shape[1]
        self.player = Agent(self.state_size, self.action_size, cls='player', seed=seed)
        self.opponent = Agent(self.state_size, self.action_size, cls='opponent', seed=seed)
        self.criteria = criteria
        
        self.score_record = Scores(2, 100)
        
        
    
    
    def reset_env(self, train=True):
       self.env_info = self.env.reset(train_mode=train)[self.brain_name]
    
    def run_episode(self, mode=0):
        """
        Runs one full episode

        Parameters
        ----------
        mode : int, optional
            Whether the episod is for training or evaluation(0:Evaluation, 1:Training). The default is 0.

        Returns
        -------
        score : int
            The total achived score in an episode.

        """
        done = [False,False]
        score = [0,0] 
        
        self.reset_env(train=mode)
        while not done[0]:
            state = self.env_info.vector_observations                        # get the current state
            player_action = self.player.act(state[0], mode)                  # Select an action for agent 1 (player)
            opponent_action = self.opponent.act(state[1], mode)              # Select an action for agent 2 (opponent)
            action = np.concatenate((player_action, opponent_action), axis=0)
            self.env_info = self.env.step(action)[self.brain_name]           # send the action to the environment
            next_state = self.env_info.vector_observations                   # get the next state
            reward = self.env_info.rewards                                   # get the reward
            done = self.env_info.local_done                                  # see if episode has finished
            
            if mode == 1:
                self.player.step(state[0], player_action, opponent_action, reward[0], next_state[0], done[0])
                self.opponent.step(state[1], opponent_action, player_action, reward[1], next_state[1], done[1])
            
            score = [sum(x) for x in zip(score, reward)]
            
        return score
    
    
    def run_evaluation_episode(self):
        score = self.run_episode(mode=0)
        return score
    
    def evaluate(self, runs=100):
        """
        Generates episodes to evaluate the current model

        Parameters
        ----------
        runs : int, optional
            Number of episodes to use in evaluation. The default is 100.
        """
        score_record = Scores(2,100)
        
        print('Evaluation in progress...')
        for i in range(runs):
            score = self.run_evaluation_episode()
            score_record.add(score)
            
        ave_score = score_record.mean()
        
        print('System evaluated with an average score of {} in {} runs'.format(ave_score, runs))
        
        
    def run_training_episode(self):
        score = self.run_episode(mode=1)
        return score
        
    def train(self, max_episodes= 2000):
        """
        Generates episodes and trains the agent until the score criteria is met

        Parameters
        ----------
        max_episodes : int, optional
            Maximum episodes to generate. The default is 1000.

        Returns
        -------
        success (bool): Whether the criteria was reached during the training or not
        """
        success = False
        i_episode = 0
        
        print('Training in progress...')
        for i in range(max_episodes):
            score = self.run_training_episode()

            self.score_record.add(score)
            
            i_episode += 1

            if i_episode%100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, self.score_record.max_average_Score()))
                
            if i_episode>100:
                if self.score_record.max_average_Score()>self.criteria:
                    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, self.score_record.max_average_Score()))
                    success = True
                    break

        if success:
            print('Criteria reached after {} episodes'.format(i_episode))
        else:
            print('Failed to reach Criteria after {} episodes'.format(i_episode))

        self.plot_training_progress()
        return success
      
     
    def plot_training_progress(self):
        """Plots the recorded scores achieved in the training phase"""
        if self.score_record:
            self.score_record.plot()
            plt.ylabel('Average score')
            plt.xlabel('Episode')
            plt.title('Average score for last 100 episodes in the training phase')
        else:
            print('No progress made yet...')
            
    
    def reset_records(self):
        """Resets all the recorded scores"""
        self.score_record.reset()
       
       
    def reset_model(self):
        self.player.reset_models()
        self.opponent.reset_models()


    def save_model(self):
        """
        saves the current model
        """
        try:
            self.player.save_model()
            self.opponent.save_model()
            print('Model saved successfully')
            return 1
        except:
            print('Failed to save model')
            return 0
            
            
    def load_model(self):
        """
        Loads a pre-trained model
        
            Parameters:
                file_name (str): Path to the saved model
        """
        try:
            self.player.load_model()
            self.opponent.load_model()
            print('Model loaded successfully')
            return 1
        except:
            print('Failed to load model')
            return 0


    def close_env(self):
        """Closes the unity environment"""
        self.env.close()
        
        
class Scores(object):
    def __init__(self, number_of_agents, window_size=100):
        self.size = number_of_agents
        self.window_size = window_size
        
        self._record = [[] for _ in range(self.size)]
        self._window = [deque(maxlen=self.window_size) for _ in range(self.size)]
        self._average = [[] for _ in range(self.size)]
        
    def add(self, score):
        assert(len(score) == self.size)
        
        for i in range(self.size):
            self._record[i].append(score[i])
            self._window[i].append(score[i])
            self._average[i].append(np.mean(self._window[i]))
        
    def mean(self):
        return [np.mean(_record) for _record in self._record]

    def max_index(self):
        return np.argmax([x[-1] for x in self.average])

    def max_average_Score(self):
        return np.max([x[-1] for x in self.average])
    
    def reset(self):
        self._record = [[] for _ in range(self.size)]
        self._window = [deque(maxlen=self.window_size) for _ in range(self.size)]
        self._average = [[] for _ in range(self.size)]
        
    def plot(self):
        """Plots the recorded scores achieved in the training phase"""
        if self._average:
            score_average = self._average[self.max_index()]
            plt.plot(score_average)
            plt.ylabel('Average score')
            plt.xlabel('Episode')
            plt.title('Average score for last 100 episodes in the training phase')
        else:
            print('No progress made yet...')

    @property
    def record(self):
        return self._record
    
    @property
    def window(self):
        return self._window
    
    @property
    def average(self):
        return self._average
        
        