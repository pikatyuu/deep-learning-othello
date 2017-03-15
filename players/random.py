import random

class Random():
    def __init__(self, name="Random"):
        self.name = name

    def action(self, game):
        return game.play(random.choice(game.movable))

    def game_finished(self, game):
        pass

    def all_game_finished(self):
        pass
