from othello import Othello

class OthelloManager():
    def __init__(self, p1, p2, loop, draw_game=False, draw_winner=False, draw_result=False, draw_probability=False):
        self.p1 = p1
        self.p2 = p2
        self.loop = loop
        self.draw_game = draw_game
        self.draw_winner = draw_winner
        self.draw_probability = draw_probability
        self.draw_result = draw_result
        self.result_stack = {"Before": 0, "After": 0, "Draw": 0}
        self.current_game = 0

    def start(self):
        while self.current_game != self.loop:
            game = Othello()
            while game.result == None:
                if self.draw_game:
                    print("")
                    game.draw()
                    print(game.movable)
                    print(f"now turn: {game.turn + 1}")
                    print("")
                if len(game.movable) == 0:
                    game.turn += 1
                    game.update_movable()
                    game.is_finished()
                    if self.draw_game:
                        print("--// pass turn //--")
                    continue
                if game.turn % 2 == 0:
                    y_x = self.p1.action(game)
                else:
                    y_x = self.p2.action(game)
                if self.draw_game:
                    print(f"put to {y_x}")
            self.p1.game_finished(game)
            self.p2.game_finished(game)
            if self.draw_game:
                print("")
                print("--// finished //--")
                game.draw()
                print(f"ended on {game.turn + 1}turn")
                print("")
            if self.draw_winner:
                print(f"{self.current_game + 1}: {self.check_winner(game)} ({game.result})")
            if self.draw_probability:
                print(f"{self.p1.name}: {game.result_probability['before']}%, {self.p2.name}: {game.result_probability['after']}%")
            self.result_stack[game.result] += 1
            self.current_game += 1
            if self.current_game % 10000 == 0 and self.current_game != self.loop:
                # temporal save
                self.p1.all_game_finished()
        self.p1.all_game_finished()
        self.p2.all_game_finished()
        if self.draw_result:
            probability = {"Before": round(self.result_stack["Before"] / self.loop * 100),"After": round(self.result_stack["After"] / self.loop * 100), "Draw": round(self.result_stack["Draw"] / self.loop * 100)}
            print(f"{self.p1.name}: {self.result_stack['Before']}, {self.p2.name}: {self.result_stack['After']}, Draw: {self.result_stack['Draw']}")
            print(f"{self.p1.name}: {probability['Before']}%, {self.p2.name}: {probability['After']}%, Draw: {probability['Draw']}%")

    def check_winner(self, game):
        if game.result == "Before":
            return self.p1.name
        elif game.result == "After":
            return self.p2.name
        else:
            return "Draw"
