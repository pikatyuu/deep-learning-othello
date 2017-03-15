import os.path as path
import random
import copy
from collections import deque
import numpy as np
from chainer import Variable, optimizers, serializers, Chain
import chainer.functions as F
import chainer.links as L

class Model(Chain):
    def __init__(self):
        super(Model, self).__init__(
            l1 = L.Linear(64, 100, initialW=random.random() * 0.02 + 1e-7),
            l2 = L.Linear(100, 200, initialW=random.random() * 0.02 + 1e-7),
            l3 = L.Linear(200, 400, initialW=random.random() * 0.02 + 1e-7),
            l4 = L.Linear(400, 64, initialW=random.random() * 0.02 + 1e-7)
        )

    def __call__(self, x, dropout=True):
        h = F.dropout(F.leaky_relu(self.l1(x)), train=dropout, ratio=0.2)
        h = F.dropout(F.leaky_relu(self.l2(h)), train=dropout, ratio=0.5)
        h = F.dropout(F.leaky_relu(self.l3(h)), train=dropout, ratio=0.5)
        return self.l4(h)

    def get_loss(self, x, t):
        return F.mean_squared_error(x, t)


class DQN():
    """
    turn (rqeuired parameter): set string "Before" or "After"
    DQN: switch whether to learn or not
    use_ER: if True, use Experience Replay
    use_target_q: if True, use target Q
    name: this player's name
    epsilon: default is 1, it will be used to the epsilon-greedy
    model_file: the file for loading and saving self.model
    optimizer_file: the file for loading and saving self.optimizer
    """
    def __init__(self, turn, DQN=True, name="DeepQ_learning", epsilon=1, model_file="model", optimizer_file="optimizer"):
        self.turn = turn
        self.DQN = DQN
        self.name = name
        self.e = epsilon
        self.gamma = 0.95
        self.last_state = None
        self.last_action = None

        # Experience Replay
        self.experience_memory_size = 1000
        self.experience_memory = deque(maxlen=self.experience_memory_size)
        self.replay_mini_batch_size = 200
        self.learning_limit = 500

        # Target Q
        self.model = Model()
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = optimizers.MomentumSGD()
        self.optimizer.setup(self.model)

        # files
        self.model_file = "data/deep_q_learning/" + model_file
        self.optimizer_file = "data/deep_q_learning/" + optimizer_file
        self.load_model()
        self.load_optimizer()

        self.count = 0
        self.random_count = 0

    def action(self, game):
        if not self.DQN:
            self.count += 1
            return game.play(self.not_learning(game))
        if len(self.experience_memory) < self.experience_memory_size // 2:
            y, x = game.play(random.choice(game.movable))
        else:
            y, x = game.play(self.policy(game))
        s = game.parse()
        a = y * 8 + x
        # win: 1, lose: -1, draw or continue: 0
        r = 0 if game.result == None else 1 if game.result == self.turn else -1
        fs = game.parse()
        terminal = 0 if game.result is None else 1
        self.store_experience(s, a, r, fs, terminal)
        self.last_state = s
        self.last_action = a
        return (y, x)

    def not_learning(self, game):
        input_data = np.array([game.parse()], dtype=np.float32)
        pred = self.model(input_data, dropout=self.DQN)
        pos = np.argmax(pred.data, axis=1)
        y = pos // 8
        x = pos % 8
        clone = game.clone()
        if not clone.can_play((y, x)):
            self.random_count += 1
            y, x = random.choice(game.movable)
        return y, x

    def store_experience(self, s, a, r, fs, terminal):
        if self.experience_memory_size == len(self.experience_memory):
            temp = np.where(np.array(self.experience_memory).T[2]==0)[0]
            if temp.size < self.experience_memory_size *  2 // 3:
                self.experience_memory.popleft()
            else:
                self.experience_memory.remove(self.experience_memory[temp[0]])
        self.experience_memory.append([s, a, r, fs, terminal])

    def update_target_model(self):
        self.target_model = copy.deepcopy(self.model)

    def policy(self, game):
        input_data = Variable(np.array([game.parse()], dtype=np.float32))
        pred = self.model(input_data, dropout=self.DQN)
        pos = np.argmax(pred.data, axis=1)
        y = pos // 8
        x = pos % 8
        if random.random() < self.e:
            y, x = random.choice(game.movable)
        else:
            self.count += 1
        if 0.2 < self.e: self.e -= 1e-3
        clone = game.clone()
        if not clone.can_play((y, x)):
            self.random_count += 1
            for y, x in game.movable:
                movable = y * 8 + x
            self.store_experience(game.parse(), pos, -1, game.parse(), 0)
            y, x = random.choice(game.movable)
        self.last_state = game.parse()
        self.last_action = y * 8 + x
        self.update_target_model()
        return y, x

    def learn(self, s, a, r, fs, terminal):
        """
        s : two-dimensional array
        a : one-dimensional array
        r : one-dimensional array
        fs: two-dimensional array
        """
        max_qs = np.max(self.target_model(np.array(list(fs), dtype=np.float32), dropout=self.DQN).data, axis=1)

        y = self.model(np.array(list(s), dtype=np.float32), dropout=self.DQN)
        t = copy.deepcopy(y)
        for i in range(len(max_qs)):
            if terminal[i]:
                t.data[i][a[i]] = r[i]
            else:
                t.data[i][a[i]] = r[i] + self.gamma * max_qs[i]
        self.model.cleargrads()
        loss = self.model.get_loss(y, t)
        loss.backward()
        print(loss.data)
        self.optimizer.update()
        if loss.data < 1e-5:
            return True
        else:
            return False

    def ER(self):
        "Experience Replay"
        experience_memory = np.array(self.experience_memory)
        perm = np.random.permutation(experience_memory)
        # for i in range(0, len(perm), self.replay_mini_batch_size):
        batch = perm[0:self.replay_mini_batch_size].T
        s = batch[0]
        a = batch[1]
        r = batch[2]
        fs = batch[3]
        terminal = batch[4]
        complete = False
        i = 0
        while not complete:
            i += 1
            complete = self.learn(s, a, r, fs, terminal)
            if self.learning_limit < i: break
        self.update_target_model()

    def game_finished(self, game):
        # if self.count:
            # print(round(self.random_count / self.count * 100))
        self.random_count = 0
        self.count = 0
        if not self.DQN: return

        if game.result == self.turn:
            self.store_experience(self.last_state, self.last_action, 1, game.parse(), 1)
        elif game.result != "Draw":
            self.store_experience(self.last_state, self.last_action, -1, game.parse(), 1)
        else:
            self.store_experience(self.last_state, self.last_action, 0, game.parse(), 1)
        self.last_state = None
        self.last_action = None
        if not (self.experience_memory_size // 2 <= len(self.experience_memory)): return
        self.ER()

    def all_game_finished(self):
        "when finished all game, run this."
        self.save_model()
        self.save_optimizer()

    def load_model(self):
        if path.exists(self.model_file):
            if path.getsize(self.model_file) == 0: return
            serializers.load_npz(self.model_file, self.model)
        else:
            with open(self.model_file, "w") as f:
                f.write("")

    def load_optimizer(self):
        if path.exists(self.optimizer_file):
            if path.getsize(self.optimizer_file) == 0: return
            serializers.load_npz(self.optimizer_file, self.optimizer)
        else:
            with open(self.optimizer_file, "w") as f:
                f.write("")

    def save_model(self):
        serializers.save_npz(self.model_file, self.model)

    def save_optimizer(self):
        serializers.save_npz(self.optimizer_file, self.optimizer)
