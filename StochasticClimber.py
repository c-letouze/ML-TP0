#! /usr/bin/env python3
# Coraline Letouzé
# 19 Sept 2020
# Machine learning for physicists - practical work 0
# Exercice 2: Stochastic climber

from math import exp, sqrt, pi, atan
from numpy import random
import pickle
import pandas as pd


class Agent:
    """
    A simple class which defines an `agent`, some decision making 
    entity which wants to maximize its own reward.
    """    
    def __init__(self, utility=0, position=(0.0,0.0),
                 utility_func = None, step_size = 1.0):
        """
        The initialization function gets called by the interpreter 
        every time a new object of the Agent class gets created.
            
        :param utility: Agent's current utility value.
        :param utility_func: Function with which to evaluate the utility
        of the agent. The funciton `utility_func` takes an (x,y) coor-
        dinate and reports a utility.
        :param position: Agent's current position (x,y)    
        """
        self.utility = utility
        self.position = position
        self.utility_func = utility_func
        self.history = dict(position = [], utility = [])
        self.step_size = step_size
    
    def __repr__(self):
        """
        This function gets called whenever the interpreter needs a 
        "formal" string representation (error logs, etc.)
        """
        return "Agent<p:%s, u:%s>" %(self.position, self.utility)
    
    def __str__(self):
        """
        This function gets called by the interpreter whenever there 
        needs to be a "user friendly"  string description of the 
        Agent object (e.g. `print(Agent())`).
        """
        return "Agent @ %s [u=%s]" % (self.position, self.utility)
    
    def record_state(self):
        """
            Store the current state of the agent in a queue.
        """
        self.history['position'].append(self.position)
        self.history['utility'].append(self.utility)
    
    def move_up(self):
        """
            Increment the Y coordinate of the agent.
        """
        self.record_state()
        self.position = (self.position[0], self.position[1]+self.step_size)
        self.utility = self.evaluate_utility()
        
    def move_down(self):
        """
        Decrement the Y coordinate of the agent.
        """
        self.record_state()
        self.position = (self.position[0], self.position[1]-self.step_size)
        self.utility = self.evaluate_utility()
        
    def move_left(self):
        """
        Decrement the X coordinate of the agent
        """
        self.record_state()
        self.position = (self.position[0]-self.step_size,
                         self.position[1])
        self.utility = self.evaluate_utility()
        
    def move_right(self):
        """
        Increment the X coordinate of the agent
        """
        self.record_state()
        self.position = (self.position[0]+self.step_size,
                         self.position[1])
        self.utility = self.evaluate_utility()
        
    def evaluate_utility(self, offset=(0.0, 0.0)):
        """
        Get the utility function relative to the current agent location.
        Optinally, evaluate at some offset from the agent.
            
        :param offset: Tuple of (x,y) coordinates with which to offset 
        the agent position to evaluate the utility.
        :return: A numeric value for the utility at the evaluated point
        """
        if self.utility_func == None:
            return 0   # Simple Agent with simple utility
        else:
            return self.utility_func((self.position[0]+offset[0],
                                      self.position[1]+offset[1]))

        
class StochasticClimber(Agent):    # Inherit from class Agent
    """
    The agent tries to find a global maximum for its utility via 
    a stochastic method.
    """

    def slope(self, dir):
        """ Return the slope of the utility function at the current position and in the specified direction dir=(x, y). """
        offset = (self.step_size*dir[0], self.step_size*dir[1])
        return (self.evaluate_utility(offset)-self.utility) / self.step_size

    def stochastic_slope(self, dir):
        return random.normal(loc=self.slope(dir),
                             scale=1/self.step_size)

    def stepper(self, opt_direction):
        """ 
        Move the agent on the grid in the chosen 
        direction 'opt_direction'. 
        """
        if opt_direction == 'North':
            self.move_up()
        elif opt_direction == 'East':
            self.move_right()
        elif opt_direction == 'South':
            self.move_down()
        elif opt_direction == 'West':
            self.move_left()
        else:
            pass
        return None
        
    def climb(self, steps):
        """ 
        Move the agent according to the greatest slope 
        but with stochastic perturbations. 
        """
        for step in range(steps):
            directions = {'East': self.stochastic_slope((1, 0)),
                          'West': self.stochastic_slope((-1, 0)),
                          'North': self.stochastic_slope((0, 1)),
                          'South': self.stochastic_slope((0, -1)),
                          'Here': self.stochastic_slope((0, 0))}   
            opt_direction = max(directions, key=directions.get)
            self.stepper(opt_direction)
        return None
            

def gaussian_utility(position, mean=(0.0,0.0), sig=1.0):
    """
    A simple utility function, essentially just an iid  multivariate 
    normal.
    """
    # Get the variance
    v = sig**2
    # Calculate distance from mean
    d = (position[0] - mean[0], position[1] - mean[1])
    dTd = d[0]**2/v + d[1]**2/v
    # Scaling
    scale = 1/(2*pi*v)
    # Final computation
    utility = scale * exp(-0.5 * dTd)
    utility += 1e-12   
    return utility

def main():

    # Define some parameters for the utility function    
    mountain_range = lambda p : gaussian_utility(p, mean=(2.0,2.0), sig = 0.4) + \
                                gaussian_utility(p, mean=(2.0,3.0), sig = 0.5) + \
                                gaussian_utility(p, mean=(1.0,1.0), sig = 0.5) + \
                                gaussian_utility(p, mean=(3.0,1.0), sig = 0.5)

    yves = StochasticClimber(utility_func=mountain_range, step_size=0.05)
    yves.climb(steps=40000)

    pickle_io = False
    if pickle_io:
        with open('yves.p', 'wb') as pickle_file:
            pickle.dump(yves.history, pickle_file)

            with open('yves.p', 'rb') as pickle_file:
                yves_history_copy = pickle.load(pickle_file)
            print (yves_history_copy)

    pandas_io = True
    if pandas_io:
        # Create a new dictionary whose keys represent the
        # headers (colums) of information that we would like to store
        # in the CSV file
        yves_csv = dict(x=[], y=[], z=[])
        yves_csv['x'] = [yves.history['position'][i][0] for i in
                         range(0, len(yves.history['position']))]
        yves_csv['y'] =  [yves.history['position'][i][1] for i in
                         range(0, len(yves.history['position']))]
        yves_csv['z'] = yves.history['utility']

        # Create a DataFrame object from a dictionary
        df = pd.DataFrame.from_dict(yves_csv)
        # Write to CSV format
        df.to_csv("yves.csv", sep=",", header=True, index=False)
        
    return None

main()
