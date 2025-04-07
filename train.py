import torch
import torch.nn.functional as F
# import torch.nn as nn
from PIL import Image
import numpy as np
import os


def load_and_preprocess(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((128, 128))
    image = torch.tensor(np.array(image)/255.0)
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

class Linear:
    def __init__(self, x, y, bias=True):
        self.w = torch.randn((x, y))
        if bias:
            self.b = torch.zeros(y)

    def __call__(self, x):
        out = x @ self.w + self.b
        return out

class Identification:
    def __init__(self):
        self.layer1 = Linear(128*128*3, 200)
        self.layer2 = Linear(200, 200)
        self.layer3 = Linear(200, 2)

    def __call__(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        logits = self.layer3(x)

        logits = logits - logits.max(dim=1, keepdim=True).values
        counts = logits.exp()
        probs = F.softmax(logits, dim=1)
        print(probs)

        loss = -probs[range(0, probs.shape[0]), lables].log().mean()
        # loss = F.cross_entropy(logits, lables)
        print(loss)

model = Identification()

data = data.view(60, 128*128*3)
print(lables)
model(data)