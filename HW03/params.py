class Params:
    def __init__(self):
        self.gamma = 0.95
        self.epsilon = 0.1
        self.alpha = 0.7
        self.k = 1e-4
        self.n = 5
        self.runs = 20

        self.methods = ['Dyna-Q', 'Dyna-Q+', 'Modified-Dyna-Q+']
        self.steps = 3000
        self.max_steps = 100
        self.changing_steps = 1000