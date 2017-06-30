from multiprocessing import Value


class CustomCounter:
    def __init__(self, max_values):
        self.max_values = max_values
        self.counters = {}
        for label, _ in max_values.items():
            self.counters[label] = Value('i', 0)

    def update(self, label, step=1):
        counter = self.counters[label]
        with counter.get_lock():
            counter.value += step
        self.print_status()

    def print_status(self):
        individual_labels = []
        for l, c in self.counters.items():
            if self.max_values[l] == 0:
                s = '{}: {:8d}'.format(l, c.value)
            else:
                s = '{}: {:3.2f}%'.format(l, 100*c.value/self.max_values[l])
            individual_labels.append(s)
        print('\t\t'.join(individual_labels), end='\r')
