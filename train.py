import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import numpy as np
import os

# CONVOLUTIONAL LAYER FORMULA: [(W−K+2P)/S]+1

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

data = torch.permute(data, (0, 3, 1, 2)).contiguous()

batch_size = 4
n_hidden = 200

# class Linear:
#     def __init__(self, x, y, bias=True):
#         self.bias = bias
#         self.w = torch.randn((x, y))
#         if self.bias:
#             self.b = torch.zeros(y)
#
#     def __call__(self, x):
#         out = x @ self.w + self.b if self.bias else x @ self.w
#         return out
#
#     def parameters(self):
#         if self.bias:
#             return [self.w] + [self.b]
#         else:
#             return [self.w]

# class BatchNorm:
#     def __init__(self, dim, eps=1e-5, momentum=0.1):
#         self.training = True
#         self.eps = eps
#         self.momentum = momentum
#
#         self.gamma = torch.ones(dim)
#         self.beta = torch.zeros(dim)
#
#         self.running_mean = torch.zeros(dim)
#         self.running_var = torch.ones(dim)
#
#     def __call__(self, x):
#         if self.training:
#             x_mean = x.mean(0, keepdim=True)
#             x_var = x.var(0, keepdim=True)
#
#         else:
#             x_mean = self.running_mean
#             x_var = self.running_var
#
#         xhat = (x - x_mean) / torch.sqrt(x_var + self.eps)
#         self.out = self.gamma * xhat + self.beta
#
#         if self.training:
#             with torch.no_grad():
#                 self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x_mean
#                 self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x_var
#
#         return self.out
#
#     def parameters(self):
#         return [self.gamma] + [self.beta]

class Identification(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1)


        self.fc = nn.Sequential(nn.Linear(1*124*124, n_hidden),
                                nn.Tanh(),
                                nn.BatchNorm1d(n_hidden),
                                nn.Linear(n_hidden, len(classes), bias=False))


        # self.layers = [
        #     Linear(data.shape[1], n_hidden),
        #     BatchNorm(n_hidden),
        #     Linear(n_hidden, n_hidden),
        #     BatchNorm(n_hidden),
        #     Linear(n_hidden, n_hidden),
        #     BatchNorm(n_hidden),
        #     Linear(n_hidden, len(classes), bias=False)
        # ]

    def forward(self, x, targets=False):
        # for p in self.parameters():
        #     p.requires_grad = True

        # for layer in self.layers:
        #     x = layer(torch.sigmoid(x))

        # x = torch.permute(x, (0, 3, 1, 2)).contiguous() # Since nn.conv2d need the channel dimension to be the second dimension
        x = self.conv1(x)
        x = self.conv2(x)
        A, B, C, D = x.shape
        x = x.view(A, -1)
        logits = self.fc(x)
        # logits = logits - logits.max(dim=1, keepdim=True).values
        # counts = logits.exp()
        probs = F.softmax(logits, dim=1)
        # probs = counts / counts.sum(dim=1, keepdim=True)

        if targets is not False:
            # loss = -probs[range(0, probs.shape[0]), targets].log().mean()
            loss = F.cross_entropy(logits, targets)
            return loss, probs
        else:
            return probs

    def identify(self, picture):
        # for layer in self.layers:
        #     layer.training = False

        model.eval()
        picture = load_and_preprocess(picture)
        picture = torch.stack((picture,), dim=0)
        A, B, C, D = picture.shape
        # picture = picture.view(A, B*C*D)
        picture = torch.permute(picture, (0, 3, 1, 2)).contiguous()
        probs = self(picture)
        idx = torch.argmax(probs)
        result = classes[idx]
        model.train()
        return result

    # def parameters(self):
    #     out = []
    #     for layer in self.layers:
    #         out = out + layer.parameters()
    #     return out

model = Identification()


max_iter = 500
lr = 0.01
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
for _ in range(max_iter):

    batch = torch.randint(0, data.shape[0], (batch_size,))
    loss, probs = model(data[batch], lables[batch])
    print(loss)

    # for p in model.parameters():
    #     p.grad = None
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # for p in model.parameters():
    #     p.data -= lr * p.grad
    optimizer.step()


test = 'data/test/caries/wc45.jpg'

print(model.identify(test))













