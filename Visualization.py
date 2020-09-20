#!/usr/bin/env python3
# Coraline Letouzé
# 20 Sept 2020
# Machine-learning for physicists: TP0 Introduction to Python
# Visualization with matplotlib

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Load the CSV dataset
yvesDF = pd.read_csv('yves.csv')

plot_type = 'path'
#one of: path, histogram, position_to_tallest, deviation_to_tallest

if plot_type == 'path':

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(yvesDF['x'], yvesDF['y'], yvesDF['z'], 'r', linewidth=0.5,
            alpha=0.75, label='Yves')
    plt.xlim((-1, 4))
    plt.ylim((-1, 4))
    ax.view_init(30, -45)

    # Axe labels
    ax.set_xlabel('West-East', fontsize=16)
    ax.set_ylabel('North-South', fontsize=16)
    ax.set_zlabel('Height', fontsize=16)

    plt.savefig('StochasticClimber_path.pdf', dpi=300,
                orientation='landscape', bbox_inches='tight')

if plot_type == 'histogram':
    
    plt.figure(figsize=(20, 10))
    
    # Histogram of X locations
    plt.subplot(121)
    plt.hist(yvesDF['x'], bins=30)
    plt.xlabel('West-East Position', fontsize=16)
    plt.ylabel('Steps at Position', fontsize=16)
    plt.title('Dwell Time, Horizontal', fontsize=16)
    
    # Histogram of Y locations
    plt.subplot(122)
    plt.hist(yvesDF['y'], bins=30)
    plt.xlabel('South-North Position', fontsize=16)
    plt.ylabel('Steps at Position', fontsize=16)
    plt.title('Dwell Time, Vertical', fontsize=16)

    plt.savefig('StochasticClimber_histogram.pdf', dpi=300,
                orientation='landscape', bbox_inches='tight')

if plot_type == 'position_to_tallest':
    
    maxstep = len(yvesDF['x'])
    plt.figure(figsize=(20, 10))
    plt.plot([0, maxstep], [2, 2], 'k', label='Tallest Peak')
    plt.plot(yvesDF['x'], linewidth=0.5, label='West-East Position',
             alpha=0.7)
    plt.plot(yvesDF['y'], linewidth=0.5, label='South-North Position',
             alpha=0.7)
    plt.xlim([1, maxstep])
    plt.xlabel('Step Count', fontsize=18)
    plt.ylabel('Position', fontsize=18)
    plt.legend(loc=4, fontsize=18)
    plt.savefig('StochasticClimber_position.pdf', dpi=300,
                orientation='landscape', bbox_inches='tight')

if plot_type == 'deviation_to_tallest':
    
    maxstep = len(yvesDF['x'])
    d = np.sqrt((yvesDF['x']-2)**2 + (yvesDF['y']-2)**2)
    plt.figure(figsize=(20, 10))
    plt.plot(d, linewidth=0.5, alpha=0.7)
    plt.xlim([1, maxstep])
    plt.ylim([0, 3])
    plt.xlabel('Step Count', fontsize=18)
    plt.ylabel('Distance from Tallest Peak', fontsize=18)
    plt.savefig('StochasticClimber_distance_tallest.pdf', dpi=300,
                orientation='landscape', bbox_inches='tight')
