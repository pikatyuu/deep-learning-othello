class Human():
    def __init__(self, name="Human"):
        self.name = name

    def action(self, game):
        safe_input = False
        while not safe_input:
            pos = input("choose a position: ")
            if pos == "draw":
                game.draw()
            elif pos == "exit":
                import sys
                sys.exit()
            elif pos == "movable":
                print(game.movable)
            elif len(pos) == 2:
                clone = game.clone()
                pos = tuple(map(int, tuple(pos)))
                if clone.can_play(pos):
                    safe_input = True
                else:
                    print("// Error: Can't put it down //")
            else:
                print("Error: Invaild input")
        return game.play(pos)

    def game_finished(self, game):
        pass

    def all_game_finished(self):
        pass
