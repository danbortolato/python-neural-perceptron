from perceptron import Perceptron

net = Perceptron()
Train = [
    {'input': [0, 0], 'output': [0]},
    {'input': [0, 1], 'output': [1]},
    {'input': [1, 0], 'output': [1]},
    {'input': [1, 1], 'output': [0]}
]
net.train(Train)
print(f'0 xor 0: {int(round(net.predict([0, 0])[0], 0))}')
print(f'0 xor 1: {int(round(net.predict([0, 1])[0], 0))}')
print(f'1 xor 0: {int(round(net.predict([1, 0])[0], 0))}')
print(f'1 xor 1: {int(round(net.predict([1, 1])[0], 0))}')
