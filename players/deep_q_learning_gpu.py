import os.path as path
import random
import copy
import numpy as np
from chainer import Variable, optimizers, Chain, serializers, cuda, functions as F, links as L

gpu_device = 0
cuda.get_device(gpu_device).use()
xp = cuda.cupy

class Model(Chain):
    def __init__(self):
        super(Model, self).__init__(
            l1 = L.Linear(64, 100),
            l2 = L.Linear(100, 200),
            l3 = L.Linear(200, 400),
            l4 = L.Linear(400, 64)
        )

    def __call__(self, x, dropout=True):
        h = F.dropout(F.leaky_relu(self.l1(x)), train=dropout, ratio=0.2)
        h = F.dropout(F.leaky_relu(self.l2(h)), train=dropout, ratio=0.5)
        h = F.dropout(F.leaky_relu(self.l3(h)), train=dropout, ratio=0.5)
        return self.l4(h)

    def get_loss(self, x, t):
        return F.mean_squared_error(x, t)


class DeepQ_learning():
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
    def __init__(self, turn, DQN=True, use_ER=True, name="DeepQ_learning", epsilon=1, model_file="product-model", optimizer_file="product-optimizer"):
        self.turn = turn
        self.DQN = DQN
        self.use_ER = use_ER
        self.name = name
        self.e = epsilon
        self.gamma = 0.95
        self.last_state = None
        self.last_action = None

        # Experience Replay
        self.experience_memory = []
        self.experience_memory_size = 10000
        self.replay_mini_batch_size = 500

        # Target Q
        self.model = Model()
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

        # files
        self.model_file = "data/deep_q_learning/" + model_file
        self.optimizer_file = "data/deep_q_learning/" + optimizer_file
        self.load_model()
        self.load_optimizer()

    def action(self, game):
        if not self.DQN:
            return game.play(self.not_learning(game))
        s = game.parse()
        if self.use_ER:
            if len(self.experience_memory) < self.experience_memory_size // 2:
                y_x = game.play(self.store_experience(game))
            else:
                y_x = game.play(self.policy(game))
        else:
            y_x = game.play(self.policy(game))
        a = y_x[0] * 8 + y_x[1]
        r = 1 if game.result == self.turn else -1
        if game.result == None: r = 0
        fs = game.parse()
        if self.experience_memory_size < len(self.experience_memory):
            temp = np.where(np.array(self.experience_memory).T[2]==0)[0]
            if temp.size == 0:
                del self.experience_memory[0]
            else:
                del self.experience_memory[np.random.choice(temp)]
        self.experience_memory.append([s, a, r, fs])
        return y_x

    def not_learning(self, game):
        input_data = Variable(np.array([game.parse()], dtype=np.float32))
        pred = self.model(input_data)
        pred = pred.data[0]
        pos = np.argmax(pred)
        y = pos // 8
        x = pos % 8
        clone = game.clone()
        if not clone.can_play((y, x)):
            y, x = random.choice(game.movable)
        return y, x

    def store_experience(self, game):
        return random.choice(game.movable)

    def update_target_model(self):
        self.target_model = copy.deepcopy(self.model)

    def policy(self, game):
        input_data = Variable(np.array([game.parse()], dtype=np.float32))
        pred = self.model(input_data)
        pos = np.argmax(pred.data, axis=1)
        y = pos // 8
        x = pos % 8
        if random.random() < self.e: y, x = random.choice(game.movable)
        if 0.2 < self.e: self.e -= 1e-3
        i = 0
        clone = game.clone()
        while not clone.can_play((y, x)):
            self.learn(game.parse(), pos, -1, game.parse(), opt=game.movable)
            pred = self.model(input_data)
            pos = np.argmax(pred.data[0])
            y = pos // 8
            x = pos % 8
            i += 1
            # if 1000 < i:
            #     print("random")
            #     y, x = random.choice(game.movable)
            clone = game.clone()
        pos = y * 8 + x
        self.last_state = game.parse()
        self.last_action = pos
        self.update_target_model()
        return y, x

    def learn(self, s, a, r, fs, opt=None):
        if s is None or a is None or fs is None: return
        self.count += 1
        if self.count % 50 == 0:
            self.count = 0
            self.update_target_model()

        fs_y = self.target_model(np.array([fs], dtype=np.float32))
        max_q = np.max(fs_y.data, axis=1)

        s_y = self.model(np.array([s], dtype=np.float32)) # now pred
        t = copy.deepcopy(s_y)
        t.data[0][a] = r + self.gamma * max_q # fix pred
        self.model.cleargrads()
        loss = self.model.get_loss(s_y, t)
        # if opt is not None:
        #     temp = []
        #     for o in opt:
        #         o = o[0] * 8 + o[1]
        #         temp.append(s_y.data[0][o])
        #     print(temp, max_q)
        print(loss.data)
        # self.plot_y.append(loss.data)
        loss.backward()
        self.optimizer.update()

    def learn_by_minibatch(self, s_batch, a_batch, r_batch, fs_batch):
        """
        s_batch : two-dimensional array
        a_batch : one-dimensional array
        r_batch : one-dimensional array
        fs_batch: two-dimensional array
        """
        max_qs = np.max(self.target_model(np.array(list(map(np.array, fs_batch)), dtype=np.float32)).data, axis=1)

        temp = self.model(np.array(list(map(np.array, s_batch)), dtype=np.float32))
        y = F.reshape(copy.deepcopy(temp), (-1, 64))
        for i in range(len(max_qs)):
            temp.data[i][a_batch[i]] = r_batch[i] + self.gamma * max_qs[i]
        t = F.reshape(temp, (-1, 64))
        self.model.cleargrads()
        loss = self.model.get_loss(y, t)
        print(loss.data)
        # print(" | ".join(map(str, [y.data[0][a_batch[0]], r_batch[0] + self.gamma * max_qs[0], loss.data])))
        loss.backward()
        self.optimizer.update()

    def ER(self):
        "Experience Replay"
        if not self.use_ER: return
        print("---- learn ER ----")
        experience_memory = np.array(self.experience_memory)
        perm = np.random.permutation(experience_memory)
        for i in range(0, len(perm), self.replay_mini_batch_size):
            batch = perm[i:i + self.replay_mini_batch_size].T
            s_batch = batch[0]
            a_batch = batch[1]
            r_batch = batch[2]
            fs_batch = batch[3]
            settled = False
            for k in range(3000):
                self.learn_by_minibatch(s_batch, a_batch, r_batch, fs_batch)
            self.update_target_model()

    def game_finished(self, game):
        "when finished one game, run this."
        if game.result == None:
            self.learn(self.last_state, self.last_action, 0, game.parse())
        elif game.result == self.turn:
            self.learn(self.last_state, self.last_action, 1, game.parse())
        elif game.result != "Draw":
            self.learn(self.last_state, self.last_action, -1, game.parse())
        else:
            self.learn(self.last_state, self.last_action, 0, game.parse())
        self.update_target_model()
        self.last_state = None
        self.last_action = None

    def all_game_finished(self):
        "when finished all game, run this."
        # plot_x = np.arange(len(self.plot_y))
        # plt.plot(plot_x + 1, self.plot_y)
        # plt.show()
        if self.use_ER: self.ER()
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
