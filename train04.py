from perceptron import Perceptron

net = Perceptron()
Train = [
    {'input': [0, 0], 'output': [0]},
    {'input': [0, 1], 'output': [1]},
    {'input': [1, 0], 'output': [1]},
    {'input': [1, 1], 'output': [0]}
]
net.train(Train)
net.saveModel()
