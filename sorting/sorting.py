import itertools as it

class Sort:
    def __init__(self, discount = 0.9, num_of_digits = 5):
        self.discount = discount
        self.num_of_digits = num_of_digits

        self.states = [x for x in it.permutations(range(num_of_digits))]
        self.values= {s:0 for s in self.states}

    def value_iteration(self):
        for i in range(1000):
            self.old_values = self.values.copy()
            for k, v in self.old_values.items():
                if k == tuple(range(self.num_of_digits)): # Terminal Node
                    self.values[k] = 10.0
                    continue
                value = 0
                for i, j in it.combinations(range(self.num_of_digits), 2):
                    k_list = list(k)
                    k_list[i], k_list[j] = k_list[j], k_list[i]
                    value = max(value, self.discount*self.old_values[tuple(k_list)])
                self.values[tuple(k)] = value

a = Sort()
a.value_iteration()
for s in a.states:
    print s, a.values[s]
