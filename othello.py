import copy

if __name__ == "__main__":
    print("This file is module. You can run play.py file.")
class Othello():
    def __init__(self):
        self.board = [[None for i in range(8)] for i in range(8)]
        self.setup_board()
        self.turn = 0
        self.result = None
        self.sequence_count = 0
        # before is black, after is white
        self.result_probability = {"before": 0, "after": 0}
        self.movable = []
        self.update_movable()

    def setup_board(self):
        self.board[3][3] = 2
        self.board[3][4] = 1
        self.board[4][3] = 1
        self.board[4][4] = 2

    def update_movable(self):
        if self.turn % 2 == 0:
            rival_symbol = 2
        else:
            rival_symbol = 1
        pieces = []
        for i in range(len(self.board)):
            for l in range(len(self.board[i])):
                if self.board[i][l] == rival_symbol:
                    pieces.append((i, l))
        all_movable = []
        # get all piece's movable on uncondition
        for piece in pieces:
            y, x = piece
            movable = []
            if x != 0:
                movable.append((y, x - 1))
            if x != 7:
                movable.append((y, x + 1))
            if y != 0:
                movable.append((y - 1, x))
            if y != 7:
                movable.append((y + 1, x))
            if y != 0 and x != 0:
                movable.append((y - 1, x - 1))
            if y != 0 and x != 7:
                movable.append((y - 1, x + 1))
            if y != 7 and x != 0:
                movable.append((y + 1, x - 1))
            if y != 7 and x != 7:
                movable.append((y + 1, x + 1))
            all_movable.extend(movable)
        # unique
        all_movable = list(set(all_movable))
        # sort all_movable on condition
        def condition(y_x):
            y, x = y_x
            if self.board[y][x] != None: return False
            def loop(i, l, not_next=True):
                if y + i > 7 or y + i < 0 or x + l > 7 or x + l < 0:
                    return False
                if self.board[y + i][x + l] == rival_symbol:
                    tmp_i = 1 if i > 0 else -1
                    tmp_l = 1 if l > 0 else -1
                    if i == 0: tmp_i = 0
                    if l == 0: tmp_l = 0
                    return loop(i + tmp_i, l + tmp_l, not_next=False)
                elif not not_next and self.board[y + i][x + l] != None:
                    return True
                return False
            if loop(1, 0): return True
            if loop(1, 1): return True
            if loop(1, -1): return True
            if loop(0, 1): return True
            if loop(0, -1): return True
            if loop(-1, 0): return True
            if loop(-1, 1): return True
            if loop(-1, -1): return True
            return False
        all_movable = list(filter(condition, all_movable))
        if len(all_movable) == 0:
            self.sequence_count += 1
        else:
            self.sequence_count = 0
        self.movable = all_movable

    def can_play(self, y_x):
        try:
            clone = self.clone()
            clone.play(y_x)
            successed = True
        except:
            successed = False
        return successed

    def play(self, y_x):
        if not y_x in self.movable: raise
        if self.turn % 2 == 0:
            my_symbol = 1
            rival_symbol = 2
        else:
            my_symbol = 2
            rival_symbol = 1
        self.reverse(y_x, my_symbol, rival_symbol)
        self.turn += 1
        self.update_movable()
        self.is_finished()
        return y_x

    def reverse(self, y_x, my_symbol, rival_symbol):
        y, x = y_x
        def loop(i, l, not_next=True):
            nonlocal temp_array, change_amount
            if y + i == 8 or y + i == -1 or x + l == 8 or x + l == -1: return
            if self.board[y + i][x + l] == rival_symbol:
                temp_array.append((y + i, x + l))
                tmp_i = 1 if i > 0 else -1
                tmp_l = 1 if l > 0 else -1
                if i == 0: tmp_i = 0
                if l == 0: tmp_l = 0
                loop(i + tmp_i, l + tmp_l, not_next=False)
            elif not not_next and self.board[y + i][x + l] != None:
                for pos in temp_array:
                    pos_y, pos_x = pos
                    self.board[pos_y][pos_x] = my_symbol
                    change_amount += 1
            return
        change_amount = 0
        for i, l in [(1, 0), (1, 1), (1, -1), (0, 1), (0, -1), (-1, 0), (-1, 1), (-1, -1)]:
            temp_array = []
            loop(i, l)
        if change_amount == 0:
            raise
        self.board[y][x] = my_symbol

    def is_finished(self):
        if self.sequence_count < 2:
            for row in self.board:
                if None in row: return
        before = 0 # black
        after = 0 # white
        for row in self.board:
            for col in row:
                if col == 1:
                    before += 1
                else:
                    after += 1
        if after < before:
            self.result = "Before"
        elif before < after:
            self.result = "After"
        else:
            self.result = "Draw"
        self.result_probability["before"] = round(before / (8 * 8) * 100)
        self.result_probability["after"] = round(after / (8 * 8) * 100)

    def parse(self):
        parsed = []
        for row in self.board:
            for col in row:
                if col == None:
                    parsed.append(0)
                else:
                    parsed.append(col)
        return parsed

    def draw(self):
        def symbol(x):
            if x == 1: return "⚫️"
            if x == 2: return "⚪️"
            if x == None: return " "
        a = [i for i in range(8)]
        print("  ", "  ".join(list(map(str, a))))
        for i in range(len(self.board)):
            temp = list(map(symbol, self.board[i]))
            print(i, "|" + " |".join(temp) + " |")

    def clone(self):
        return copy.deepcopy(self)
