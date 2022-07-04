class Perceptron():
    def __init__(self, bias=1, hiddenNodes=4, Activation='sigmoid', Epochs=1):
        import math as m
        self.m = m
        self.Weights = []
        self.Inputs = []
        self.Targets = []
        self.Epochs = Epochs
        self.Activation = Activation
        self.hiddenNodes = hiddenNodes
        self.bias = bias

    def gradientDescent(self, n=0):
        return n * (1 - n)

    def tanh(self, n=0):
        try:
            return self.m.sinh(n) / self.m.cosh(n)
        except:
            return 0

    def sigmoid(self, n=0):
        try:
            return 1 / (1 + pow(self.m.e, -n))
        except:
            return 0

    def relu(self, n=0):
        try:
            if n > 1: n = 1
            return max([n, 0])
        except:
            return 0

    def leakyRelu(self, n=0):
        try:
            if n > 1: n = 1
            return max([n, .01])
        except:
            return 0

    def binaryStep(self, n=0):
        try:
            if n >=0: return 1
            else: return 0
        except:
            return 0

    def updateList(self, matrix=[]):
        resultMatrix = []
        for m in matrix:
            vector = m
            resultVector = []
            for v in vector:
                element = v
                if element == 0: element = .01
                if element > 1:
                    res = 0
                    if str(element).find('.') > 0:
                        temp = str(element).split('.')
                        string = temp[0].replace('-', '')
                    else:
                        string = str(element)
                    length = len(string)
                    div = int('1'+'0'.ljust(length, '0'))
                    res = element / div
                    resultVector.append(res)
                else:
                    resultVector.append(element)
            resultMatrix.append(resultVector)
        return resultMatrix

    def feedForward(self, inputs=[], target=0, epochs=1, activation='sigmoid', hiddenNodes=4):
        matrixHidden = []
        vectorHidden = []
        for j in range(hiddenNodes):
            vectorHidden.append(.001)
        matrixHidden.append(vectorHidden)

        stop = False
        output = 0
        if target != 0:
            for i in range(1, epochs+1):
                multiply = []
                for j in range(len(inputs)):
                    for x in range(len(matrixHidden)):
                        for y in range(len(matrixHidden[x])):
                            multiply.append(inputs[j] * matrixHidden[x][y])
                _sum = sum(multiply)
                if activation == 'tanh': output = round(self.tanh(_sum), 4)
                elif activation == 'sigmoid': output = round(self.sigmoid(_sum), 4)
                elif activation == 'relu': output = round(self.relu(_sum), 4)
                elif activation == 'leakyRelu': output = round(self.leakyRelu(_sum), 4)
                elif activation == 'binaryStep': output = round(self.binaryStep(_sum), 4)
                else: output = round(self.sigmoid(_sum), 4)
                error = round(abs(target - output), 4)
                if error <= .1 and stop == False:
                    self.Weights.append([inputs, matrixHidden])
                    break
                    stop = True
                for j in range(len(inputs)):
                    for x in range(len(matrixHidden)):
                        for y in range(len(matrixHidden[x])):
                            matrixHidden[x][y] += inputs[j] * self.gradientDescent(error)

        if stop == False:
            if output > target:
                for x in range(len(matrixHidden)):
                    for y in range(len(matrixHidden[x])):
                        matrixHidden[x][y] -= self.bias
            elif output < target:
                for x in range(len(matrixHidden)):
                    for y in range(len(matrixHidden[x])):
                        matrixHidden[x][y] += self.bias
            self.Weights.append([inputs, matrixHidden])

    def train(self, fit=[]):
        for f in fit:
            if 'input' in f: self.Inputs.append(f['input'])
            else: self.Inputs.append([0])
            if 'output' in f: self.Targets.append(f['output'])
            else: self.Targets.append([0])

        self.Inputs = self.updateList(self.Inputs)
        self.Targets = self.updateList(self.Targets)

        i = 0
        while i < len(self.Inputs):
            j = 0
            while j < len(self.Targets):
                try:
                    if self.Inputs[i] and self.Targets[i][j]:
                        self.feedForward(self.Inputs[i], self.Targets[i][j], self.Epochs, self.Activation, self.hiddenNodes)
                except:
                    pass
                j += 1
            i += 1

    def saveModel(self, Path='model.bin'):
        write = open(Path, 'w')
        txt = f'Weights:{self.Weights}\n'
        txt += f'Activation:{self.Activation}\n'
        txt += f'hiddenNodes:{self.hiddenNodes}'
        write.write(txt)
        write.close()

    def loadModel(self, Path='model.bin'):
        from json import loads
        from os import path

        if not path.exists(Path):
            self.saveModel()
        read = open(Path, 'r')

        for line in read:
            result = line.strip()
            vector = result.split(':')
            if vector[0] == 'Weights':
                self.Weights = loads(vector[1])
            elif vector[0] == 'Activation':
                self.Activation = vector[1]
            elif vector[0] == 'hiddenNodes':
                self.hiddenNodes = int(vector[1])
        read.close()

    def predict(self, inputs=[]):
        inputs = self.updateList([inputs])
        inputs = inputs[0]
        Outputs = []
        diff = []
        for i in range(len(self.Weights)):
            Input = self.Weights[i][0]
            Sum = 0
            for j in range(len(inputs)):
                Sum += abs(inputs[j] - Input[j])
            diff.append({'index': i, 'value': Sum})

        from math import inf
        _min = inf
        index = 0
        for i in range(len(diff)):
            if diff[i]['value'] < _min:
                _min = diff[i]['value']
                index = diff[i]['index']

        try:
            if self.Targets[0]: limit = len(self.Targets[0])
        except:
            limit = 1

        for i in range(limit):
            try:
                matrixHidden = self.Weights[index][1]
            except:
                matrixHidden = [[0]]
            multiply = []
            for j in range(len(inputs)):
                for x in range(len(matrixHidden)):
                    for y in range(len(matrixHidden[x])):
                        multiply.append(inputs[j] * matrixHidden[x][j])
            _sum = sum(multiply)
            if self.Activation == 'tanh': output = round(self.tanh(_sum), 4)
            elif self.Activation == 'sigmoid': output = round(self.sigmoid(_sum), 4)
            elif self.Activation == 'relu': output = round(self.relu(_sum), 4)
            elif self.Activation == 'leakyRelu': output = round(self.leakyRelu(_sum), 4)
            elif self.Activation == 'binaryStep': output = round(self.binaryStep(_sum), 4)
            else: output = round(self.sigmoid(_sum), 4)
            Outputs.append(output)

        return Outputs
