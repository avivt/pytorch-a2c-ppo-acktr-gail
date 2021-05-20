import sys
from contextlib import closing

import numpy as np
from io import StringIO

from gym import utils
from gym.envs.toy_text import discrete

from gym import Env, spaces
from gym.utils import seeding

R_DISTS = {
    'fixed': np.array(
    [[0.1, 0.7, 0.8, 0.3, 0.2, 0.1]]),
    'train': np.array(
    [[0.1, 0.7, 0.8, 0.3, 0.2, 0.1],
    [0.8, 0.6, 0.5, 0.4, 0.1, 0.2],
    [0.1, 0.2, 0.3, 0.4, 0.8, 0.7]]),
    'train_5': np.array(
    [[0.1, 0.7, 0.8, 0.3, 0.2, 0.1],
    [0.8, 0.6, 0.5, 0.4, 0.1, 0.2],
    [0.1, 0.2, 0.3, 0.4, 0.8, 0.7],
    [0.4, 0.3, 0.5, 0.2, 0.1, 0.6],
    [0.7, 0.5, 0.5, 0.8, 0.7, 0.5]]),
    'train_10': np.array(
    [[0.1, 0.7, 0.8, 0.3, 0.2, 0.1],
    [0.8, 0.6, 0.5, 0.4, 0.1, 0.2],
    [0.1, 0.2, 0.3, 0.4, 0.8, 0.7],
    [0.4, 0.3, 0.5, 0.2, 0.1, 0.6],
    [0.23, 0.4, 0.4, 0.12, 0.15, 0.53],
    [0.56, 0.1, 0.3, 0.34, 0.85, 0.32],
    [0.45, 0.3, 0.9, 0.1, 0.3, 0.74],
    [0.7, 0.5, 0.5, 0.8, 0.7, 0.5]]),
    'test': np.array(
    [[0.5583209669457441, 0.14378224337523915, 0.9436186104124321, 0.99726729640113, 0.10641860073659604, 0.13712139004697155],
     [0.17618200951249885, 0.7172043529729938, 0.689735439163216, 0.7550671694281562, 0.5099650818065524, 0.6977404471587776],
     [0.3362013055819313, 0.8121671725198164, 0.1851888250694611, 0.86944348797489, 0.604585793610627, 0.9865039169230189],
     [0.6797818359946657, 0.7945944740970651, 0.12047570217292702, 0.6563850015870236, 0.5000989336104524, 0.0013348156380103005],
     [0.4103331047073757, 0.9280487218285304, 0.19754741245589158, 0.445697040538063, 0.19557218747184069, 0.757959558136865],
     [0.8857398939527538, 0.9807082544782421, 0.19568902367022523, 0.8618109428909894, 0.7537628052696965, 0.5755020802558202],
     [0.13422277591429532, 0.32704191608946886, 0.7367101505242638, 0.19825736695881024, 0.7284688397238258, 0.5719342449681714],
     [0.1793736305672361, 0.23644343416025448, 0.9322506101968691, 0.10577849457987931, 0.5574994433689389, 0.2540780613402486],
     [0.3295430251271414, 0.0011226768499100759, 0.7233443144204476, 0.6005372945257655, 0.4384830804399886, 0.5967560793251007],
     [0.8074894693720496, 0.770713822989823, 0.6126076547045539, 0.4865181817061386, 0.3190944808708267, 0.2277280017207074],
     [0.6112364727005241, 0.6435965504208496, 0.932615305110968, 0.9813314024045093, 0.1351175267788669, 0.8398711729783856],
     [0.0034885040252133903, 0.20900338557209275, 0.9983213596388968, 0.13862410027645222, 0.047615355555320926, 0.21670396948714565],
     [0.36986777720992337, 0.6886350882563345, 0.3731080788125013, 0.4278845701898749, 0.31252008274709897, 0.3832351475613319],
     [0.5758268807670393, 0.09500254683953291, 0.06856998143128079, 0.7765243899751935, 0.30041287950248574, 0.646384980315658],
     [0.07957784413997893, 0.6885914851465013, 0.6891383069677297, 0.0893091454076731, 0.3329999096949209, 0.3082679785772545],
     [0.5384140668340268, 0.35102353784305695, 0.14342882147728764, 0.4935398532463082, 0.23986696417127074, 0.4496222880424673],
     [0.04889186525006006, 0.09549778029685152, 0.05981190376735568, 0.8819116757950379, 0.4108460225003686, 0.687496850325133],
     [0.34691626808070086, 0.9651637941013143, 0.31485930276677543, 0.3182555218569374, 0.894923208719056, 0.6413665885989818],
     [0.8774318168312971, 0.4940845453086953, 0.3114436112639196, 0.6202009384443135, 0.7700128564631012, 0.662164743137881],
     [0.5415387366593426, 0.9956711210883071, 0.3476905277679645, 0.5913969949568693, 0.7465454215537669, 0.09744368507672274]]
    )
}

def generate_random_rewards(size=6):
    """Generates a random assignments of reward
    :param size: number of arms
    """
    r = np.random.uniform(size=size)
    return r


class HBanditEnv(Env):
    """
    MAB env
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, n=6, arms=None, steps=10, r_dist_name="6_v0", normalize_r=True, free_exploration=0):
        self.n = 6
        self.seed()
        if arms is None and r_dist_name == "random":
            self.arms = RandArms(n=n)
        elif arms is None:
            self.arms = RandChooseArms(n=n, r_dist_name=r_dist_name)
        # elif arms is None:
        #     self.arms = ConstantArms(n=n, r_dist_name=r_dist_name)
        self.reward_range = (0, 1)
        self.t = 0
        self.steps = steps
        self.normalize_r = normalize_r
        self.free_exploration = free_exploration

        self.action_space = spaces.Discrete(self.n)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=[2 * self.steps], dtype=np.float32)

        self.s = np.zeros(2 * self.steps)
        self.lastaction = None

        self.task_id = -1

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.arms.reset()
        self.t = 0
        self.s = np.zeros(2 * self.steps)
        self.lastaction = None
        return self.s

    def step(self, a):
        r, r_rank = self.arms.eval(a)
        self.s[2 * self.t] = a
        self.s[2 * self.t+1] = r
        self.lastaction = a
        self.t += 1
        done = (self.t == self.steps)
        reward = r
        if self.t < self.free_exploration:
            reward = 0
        if self.normalize_r:
            reward = r_rank
        return self.s.copy(), reward, done, {"time": self.t}

    def render(self, mode='human'):
        print(self.arms)
        print(self.t)
        print(self.s)

    def set_task_id(self, task_id):
        self.arms.task_id = task_id
        self.arms.reset()
        self.task_id = task_id


class ConstantArms():
    def __init__(self, n=6, r_dist_name="6_v0"):
        self.arms = np.array(R_DISTS[r_dist_name])
        self.n = len(self.arms)
        # temp = self.arms.argsort()
        # ranks = np.empty_like(temp)
        # ranks[temp] = np.arange(self.n)
        inds = self.arms.argsort()
        ranks = np.empty_like(inds)
        # tmp = np.arange(self.n)
        tmp = np.zeros(self.n)
        tmp[-1] = self.n - 1
        ranks[inds] = tmp
        self.ranks = ranks / (self.n-1)
        self.task_id = -1

    def eval(self, arm):
        return self.arms[arm], self.ranks[arm]

    def reset(self):
        return


class RandChooseArms():
    def __init__(self, n=6, r_dist_name='train'):
        self.r_list = R_DISTS[r_dist_name]
        self.arms = np.array(self.r_list[np.random.randint(len(self.r_list))])
        self.n = len(self.arms)
        inds = self.arms.argsort()
        ranks = np.empty_like(inds)
        # tmp = np.arange(self.n)
        tmp = np.zeros(self.n)
        tmp[-1] = self.n-1
        ranks[inds] = tmp
        self.ranks = ranks / (self.n-1)
        self.task_id = -1

    def eval(self, arm):
        return self.arms[arm], self.ranks[arm]

    def reset(self):
        if self.task_id < 0:
            task_id = np.random.randint(len(self.r_list))
        else:
            task_id = self.task_id
        self.arms = np.array(self.r_list[task_id])
        self.n = len(self.arms)
        # temp = self.arms.argsort()
        # ranks = np.empty_like(temp)
        # ranks[temp] = np.arange(self.n)
        inds = self.arms.argsort()
        ranks = np.empty_like(inds)
        # tmp = np.arange(self.n)
        tmp = np.zeros(self.n)
        tmp[-1] = self.n - 1
        ranks[inds] = tmp
        self.ranks = ranks / (self.n-1)
        return


class RandArms():
    def __init__(self, n=6):
        self.arms = generate_random_rewards()
        self.n = len(self.arms)
        # temp = self.arms.argsort()
        # ranks = np.empty_like(temp)
        # ranks[temp] = np.arange(self.n)
        inds = self.arms.argsort()
        ranks = np.empty_like(inds)
        # tmp = np.arange(self.n)
        tmp = np.zeros(self.n)
        tmp[-1] = self.n - 1
        ranks[inds] = tmp
        self.ranks = ranks / (self.n-1)
        self.task_id = -1

    def eval(self, arm):
        return self.arms[arm], self.ranks[arm]

    def reset(self):
        self.arms = generate_random_rewards()
        self.n = len(self.arms)
        # temp = self.arms.argsort()
        # ranks = np.empty_like(temp)
        # ranks[temp] = np.arange(self.n)
        inds = self.arms.argsort()
        ranks = np.empty_like(inds)
        # tmp = np.arange(self.n)
        tmp = np.zeros(self.n)
        tmp[-1] = self.n - 1
        ranks[inds] = tmp
        self.ranks = ranks / (self.n-1)
        return
