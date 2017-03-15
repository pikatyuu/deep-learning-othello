from othello_manager import OthelloManager
from players.random import Random
from players.deep_q_learning_gpu import DeepQ_learning_GPU
from players.DQN_gpu import DQN_GPU

manager = OthelloManager(DQN_GPU("Before"), Random(), loop=100000, draw_winner=True, draw_game=False, draw_result=True, draw_probability=False)
manager.start()
