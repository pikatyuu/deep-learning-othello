from othello_manager import OthelloManager
from players.human import Human
from players.random import Random
from players.deep_q_learning import DeepQ_learning
from players.DQN import DQN

# manager = OthelloManager(DeepQ_learning("Before", DQN=False), Human(), loop=1, draw_winner=True, draw_game=True, draw_result=True)
# manager = OthelloManager(DeepQ_learning("Before", DQN=False), Random(), loop=100, draw_winner=True, draw_game=False, draw_result=True)
# manager = OthelloManager(DeepQ_learning("Before", use_ER=False), Random(), loop=100, draw_winner=True, draw_game=False, draw_result=True, draw_probability=False)

manager = OthelloManager(DQN("Before"), Random(), loop=200, draw_winner=True, draw_game=False, draw_result=True, draw_probability=False)
# manager = OthelloManager(DQN("Before", DQN=False), Random(), loop=100, draw_winner=True, draw_game=False, draw_result=True)
manager.start()
