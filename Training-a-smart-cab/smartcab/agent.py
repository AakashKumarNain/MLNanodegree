import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import pandas as pd
import math

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""


    def __init__(self, env, learning_rate, decay_rate):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
    
        self.state = None
        self.alpha = learning_rate
        self.gamma = decay_rate
        self.Qvalues = self.initQvalues()
        self.total_rewards = 0.0
        self.total_actions = 0
        self.penalty = 0.0
        self.journeys_completed = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def initQvalues(self, value=1.0):        
        qvalues = {}
        
        for waypoint in ['left', 'right', 'forward']:
            for light in ['red', 'green']:
                for i in [None, 'forward', 'left', 'right']:
                    for action in self.env.valid_actions:
                        qvalues[((waypoint, light, i), action)] = value
            
        return qvalues
              


    def getMaxQvalue(self, state):
            maxQval = 0.0
            maxvalAction = None

            for action in self.env.valid_actions:
                qval = self.Qvalues[(state, action)]
                if qval > maxQval:
                    maxvalAction = action
                    maxQval = qval
            return(maxQval, maxvalAction)        


    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (self.next_waypoint, inputs['light'], inputs['oncoming'])

        # TODO: Select action according to your policy
        
        #action = random.choice(self.env.valid_actions)    # Basic driving agent choosing a random action
        (Q, action) = self.getMaxQvalue(self.state)        # For Q-learning  

        # Execute action and get reward
        reward = self.env.act(self, action)
        
        if reward < 0.0 :
            self.penalty += reward

        if deadline >= 0 and self.env.done:
            self.journeys_completed +=1    
        
        self.total_rewards += reward
        self.total_actions += 1

        
        # TODO: Learn policy based on state, action, reward
        next_waypoint = self.planner.next_waypoint()
        inputs = self.env.sense(self)
        next_state = (next_waypoint, inputs['light'], inputs['oncoming'])
        
        (Q_prime, next_action) = self.getMaxQvalue(next_state)

        Q += self.alpha * (reward + self.gamma*Q_prime - Q) 
        self.Qvalues[(self.state, action)] = Q

        
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        
        


def run(alpha_value, gamma_value):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent,learning_rate=alpha_value, decay_rate=gamma_value)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    return a

if __name__ == '__main__':

    #alphas = [0.1, 0.2, 0.3, 0.5, 0.6, 0.9]
    #gammas = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8]

    alphas = [0.6]         # optimal alpha_value after grid search
    gammas = [0.3]         # optimal gamma_value after grid search 


    alpha_values = []
    gamma_values = []
    total_moves = [] 
    total_rewards = []
    average_penalties = []
    average_rewards = [] 
    reward_ratio = []
    penalty_ratio = []
    no_of_journey_completed = []
     
    
    for alpha_value in alphas:
        for gamma_value in gammas:
            alpha_values.append(alpha_value)
            gamma_values.append(gamma_value) 

            a = run(alpha_value, gamma_value)
            
            actions_taken = a.total_actions
            rewards_obtained = a.total_rewards
            average_reward = rewards_obtained + a.penalty
            
            total_moves.append(int(actions_taken))
            total_rewards.append(rewards_obtained)
            average_penalties.append(a.penalty)
            average_rewards.append(average_reward)
            reward_ratio.append(average_reward / actions_taken)
            penalty_ratio.append((abs(a.penalty))/ actions_taken)
            no_of_journey_completed.append(a.journeys_completed)
            

    #print no_of_journey_completed
    # Run the below code in order to generate a csv giving detailed descriptions
    # df = pd.DataFrame({'alpha' : alpha_values,
    #                     'gamma' : gamma_values,
    #                  'total_moves' : total_moves,
    #                  'total_reward' : total_rewards,
    #                  'penalty': average_penalties,
    #                   'average_reward': average_rewards,
    #                   'reward_ratio': reward_ratio,
    #                   'journeys_completed' : no_of_journey_completed,
    #                   'penalty_ratio' : penalty_ratio})
    
    # df.to_csv('smartcab_performance.csv', columns =['alpha','gamma', 'total_moves', 'total_reward', 'penalty', 'average_reward', 'reward_ratio', 'journeys_completed', 'penalty_ratio'],index=False)

