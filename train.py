import torch
import torch.nn.functional as F
# import torch.nn as nn
from PIL import Image
import numpy as np
import os


def load_and_preprocess(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((128, 128))
    image = torch.tensor(np.array(image)/255.0, dtype=torch.float)
    return image

def load_data(data_dir):
    data = []
    labels = []
    classes = sorted(os.listdir(data_dir))
    for idx, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                image = load_and_preprocess(image_path)
                data.append(image)
                labels.append(idx)

    labels = torch.tensor(labels)
    data = torch.tensor(np.array(data), dtype=torch.float)
    return data, labels, classes

image = load_and_preprocess('data/training/caries/wc3.jpg')
data, lables, classes = load_data('data/training')

data = data.view(-1, 128*128*3)

batch_size = 10
n_hidden = 200

class Linear:
    def __init__(self, x, y, bias=True):
        self.bias = bias
        self.w = torch.randn((x, y))
        if self.bias:
            self.b = torch.zeros(y)

    def __call__(self, x):
        out = x @ self.w + self.b if self.bias else x @ self.w
        return out

    def parameters(self):
        if self.bias:
            return [self.w] + [self.b]
        else:
            return [self.w]

class BatchNorm:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.training = True
        self.eps = eps
        self.momentum = momentum

        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            x_mean = x.mean(0, keepdim=True)
            x_var = x.var(0, keepdim=True)

        else:
            x_mean = self.running_mean
            x_var = self.running_var

        xhat = (x - x_mean) / torch.sqrt(x_var + self.eps)
        self.out = self.gamma * xhat + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x_var

        return self.out

    def parameters(self):
        return [self.gamma] + [self.beta]

class Identification:
    def __init__(self):
        self.layers = [
            Linear(data.shape[1], n_hidden),
            BatchNorm(n_hidden),
            Linear(n_hidden, n_hidden),
            BatchNorm(n_hidden),
            Linear(n_hidden, n_hidden),
            BatchNorm(n_hidden),
            Linear(n_hidden, len(classes), bias=False)
        ]

    def __call__(self, x, targets=False):
        for p in self.parameters():
            p.requires_grad = True

        for layer in self.layers:
            x = layer(torch.sigmoid(x))
        logits = x

        logits = logits - logits.max(dim=1, keepdim=True).values
        counts = logits.exp()
        # probs = F.softmax(logits, dim=1)
        probs = counts / counts.sum(dim=1, keepdim=True)

        if targets is not False:
            loss = -probs[range(0, probs.shape[0]), targets].log().mean()
            # loss = F.cross_entropy(logits, lables)
            return loss, probs
        else:
            return probs

    def identify(self, picture):
        for layer in self.layers:
            layer.training = False
        picture = load_and_preprocess(picture)
        picture = torch.stack((picture,), dim=0)
        A, B, C, D = picture.shape
        picture = picture.view(A, B*C*D)
        probs = self(picture)
        idx = torch.argmax(probs)
        result = classes[idx]
        return result

    def parameters(self):
        out = []
        for layer in self.layers:
            out = out + layer.parameters()
        return out

model = Identification()


max_iter = 500
lr = 0.1
for _ in range(max_iter):
    batch = torch.randint(0, data.shape[0], (batch_size,))

    loss, probs = model(data[batch], lables[batch])
    print(loss)

    for p in model.parameters():
        p.grad = None

    loss.backward()

    for p in model.parameters():
        p.data -= lr * p.grad



test = 'data/test/caries/wc45.jpg'

print(model.identify(test))













