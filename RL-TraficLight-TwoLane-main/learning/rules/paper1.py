from learning.modules import *

class OneObservOneReward:
    def __init__(self, observasi='tf', reward='wt'):
        self.observasi = observasi
        self.observasi_len = 1
        self.reward = reward
        self.max_lanes = 2
        self.duration_all_red=0
        self.duration_yellow_red=0
        self.duration_max_phase=30
        self.action_list = [
            [self.duration_max_phase/2, 0],
            [0, self.duration_max_phase/2]
        ]

    def countReward(self, junction, log) :
        return sum(list(log[junction][self.reward].values())) * (1 if self.reward == 'as' else -1)

    def observationMatix(self, junction, log):
        return list(log[junction][self.observasi].values())