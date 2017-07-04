import itertools as it
import math
import random

class Sort:
    def __init__(self, num_of_digits = 5):
        self.num_of_digits = num_of_digits
        self.reward = 10.0
        self.discount = -1

        self.states = [x for x in it.permutations(range(num_of_digits))]
        self.values= {s:0 for s in self.states}
        self.qvalues = {(s,a):0 for s in self.states for a in it.combinations(range(self.num_of_digits), 2)}

    ### Value Iteration ###
    def value_iteration(self, discount = 0.9):
        assert discount >= 0 and discount <= 1

        self.discount = discount
        for i in range(100):
            self.old_values = self.values.copy()
            for k, v in self.old_values.items():
                if k == tuple(range(self.num_of_digits)): # Terminal Node
                    self.values[k] = self.reward
                    continue
                value = 0
                for i, j in it.combinations(range(self.num_of_digits), 2):
                    k_list = list(k)
                    k_list[i], k_list[j] = k_list[j], k_list[i]
                    value = max(value, discount*self.old_values[tuple(k_list)])
                self.values[tuple(k)] = value
    def print_value_function(self):
        for s in self.states:
            print s, self.values[s]

    def print_num_of_swap(self):
        if self.discount < 0:
            print "Error : Take value_iteration step before print_num_of_swap"
            return

        for s in self.states:
            print s, int(math.log(self.values[s]/10.0, self.discount))
    ### Value Iteration End ###

    ### Sarsa Lambda ###
    def sarsa_lambda(self, num_of_episodes = 5000, epsilon = 0.1, _lambda = 0.9, learning_rate = 0.1, discount = 0.9):
        assert discount >= 0 and discount <= 1
        assert epsilon >= 0 and epsilon <= 1
        assert _lambda >= 0 and _lambda <= 1

        # Repeat episode
        for i in range(num_of_episodes):
            print i, "th Episode"
            # Initialize eligibility traces
            e = {(s,a):0 for s in self.states for a in it.combinations(range(self.num_of_digits), 2)}
            # Initialize state and action
            s = range(self.num_of_digits); random.shuffle(s); s = tuple(s)
            a = self.choose_action(s, self.qvalues, epsilon)
            while True:
                if list(s) == range(self.num_of_digits): break
                next_s, reward = self.take_action(s, a)
                next_a = self.choose_action(next_s, self.qvalues, epsilon)

                delta = reward + discount*self.qvalues[(next_s, next_a)] - self.qvalues[(s, a)]
                e[(s,a)] = e[(s,a)] + 1
                for state in self.states:
                    for action in it.combinations(range(self.num_of_digits),2):
                        self.qvalues[(state,action)] = self.qvalues[(state,action)] + learning_rate*delta*e[(state,action)]
                        e[(state,action)] = discount*_lambda*e[(state,action)]
                s, a = next_s, next_a
        print "Finished ", i, " Episodes"

    def print_qvalues(self):
        for k, v in self.qvalues.iteritems():
            s, _ = self.take_action(k[0], k[1])
            print k, s, v


                
    def take_action(self, s, a):
        assert type(s) == tuple and  len(s) == self.num_of_digits
        assert type(a) == tuple and len(a) == 2 and a[0] != a[1]
        s = list(s)
        s[a[0]], s[a[1]] = s[a[1]], s[a[0]]

        if s == range(self.num_of_digits):
            reward = self.reward 
        else:
            reward = 0

        return tuple(s), reward
    def choose_action(self, current_state, qvalues, epsilon):
        action_values = {a: qvalues[(current_state, a)] for a in it.combinations(range(self.num_of_digits), 2)}
        if random.random() > epsilon:
            return max(action_values, key=action_values.get)
        else:
            return random.choice(action_values.keys())
    ### Sarsa Lambda End ###

