import random
import os
import time
import neat
import visualize
import pickle
from bareSnake import snakeGame
import numpy as np
import math
from neat.graphs import feed_forward_layers
from neat.graphs import required_for_output
import tensorflow as tf

tf.config.run_functions_eagerly(True)

import pandas as pd

df = pd.read_csv("testData3.csv")
df.drop(columns=['Unnamed: 0'])

hld = 2

dfMunch = df.query('intention == "munch"').drop(columns=['Unnamed: 0']).reset_index(drop=True)

dfWall = df.query('intention == "wall move"').drop(columns=['Unnamed: 0']).reset_index(drop=True)

dfFood = df.query('intention == "food move"').drop(columns=['Unnamed: 0']).reset_index(drop=True)


MunchMoves = [[],[]]
WallMoves = [[],[]]
FoodMoves = [[],[]]

for i in range(len(dfMunch)):
    finalIns = []
    finalOuts = []
    ins = np.array(dfMunch.values[i][0][1:-1].split(','),dtype=float)
    outs = np.array(dfMunch.values[i][1][1:-1].split(','),dtype=int)
    for j in range(len(ins)):
        finalIns.append(ins[j])
    for j in range(len(outs)):
        finalOuts.append(outs[j])
    MunchMoves[0].append(finalIns)
    MunchMoves[1].append(finalOuts)

for i in range(len(dfFood)):
    ins = np.array(dfFood.values[i][0][1:-1].split(','),dtype=float)
    outs = np.array(dfFood.values[i][1][1:-1].split(','),dtype=int)
    FoodMoves[0].append(ins)
    FoodMoves[1].append(outs)

for i in range(len(dfWall)):
    ins = np.array(dfWall.values[i][0][1:-1].split(','),dtype=float)
    outs = np.array(dfWall.values[i][1][1:-1].split(','),dtype=int)
    WallMoves[0].append(ins)
    WallMoves[1].append(outs)


hold = 8







def find_index(array):
    s = np.array(array)
    sort_index = np.argsort(s)
    return sort_index

def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    p = neat.Population(config)

    genomes = p.population.items()
    genome1key = next(iter(p.population))
    genome = p.population[genome1key]
    print(genome1key)
    print(genome)
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    lr = .1
    batchSize = 400
    epochSize = 100

    OgAvgMunch = net.batchAcc(MunchMoves[0][0:batchSize],MunchMoves[1][0:batchSize])
    OgAvgFood = net.batchAcc(FoodMoves[0][0:batchSize], FoodMoves[1][0:batchSize])
    OgAvgWall = net.batchAcc(WallMoves[0][0:batchSize], WallMoves[1][0:batchSize])

    asdf = 9
    for k in range(epochSize):
        net.backProp_GenomeFast(MunchMoves[0][0:batchSize],MunchMoves[1][0:batchSize],lr,genome)
        #print(net.batchAcc(MunchMoves[0][0:batchSize],MunchMoves[1][0:batchSize]))

    newAvgMunch = net.batchAcc(MunchMoves[0][0:batchSize],MunchMoves[1][0:batchSize])
    asdfasdf = 95
    print("at the end\n\n i hope")
    print("the first acc is %d", OgAvgMunch)
    print("the after training acc is $d", newAvgMunch)
if __name__ == '__main__':
    #print('in start')
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'configParam2.txt')
    run(config_path)