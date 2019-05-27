import cv2
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Renderer.model import FCN
from Renderer.stroke_gen import draw_rect
from utils.tensorboard import TensorBoard

# writer = TensorBoard("../train_log/")
writer = TensorBoard("./train_log/")

# action dimension
action_dim = 4
draw_fn = draw_rect

criterion = nn.MSELoss()
net = FCN(num_input=action_dim)
optimizer = optim.Adam(net.parameters(), lr=3e-6)
batch_size = 4096

use_cuda = torch.cuda.is_available()
step = 0


def save_model():
    if use_cuda:
        net.cpu()
    torch.save(net.state_dict(), "./renderer.pkl")
    if use_cuda:
        net.cuda()
    print("saved model")


def load_weights():
    pretrained_dict = torch.load("./renderer.pkl")
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    print("loaded pretrained weights")


# load_weights()

# while step < 500000:
while step < 10000:
    net.train()
    x = []
    gt = []

    # generate ground truth data
    for i in range(batch_size):
        # _x = np.random.uniform(0, 1, action_dim)
        _x = np.array([0, 0, 0.5, 0.5])
        x.append(_x)
        gt.append(draw_fn(_x))

    x = torch.tensor(x).float()
    gt = torch.tensor(gt).float()
    if use_cuda:
        net = net.cuda()
        x = x.cuda()
        gt = gt.cuda()

    y = net(x)
    # print(gt)
    # print(y)
    optimizer.zero_grad()
    loss = criterion(y, gt)
    loss.backward()
    optimizer.step()
    print(step, loss.item())

    if step < 200000:
        lr = 1e-4
    elif step < 400000:
        lr = 1e-5
    else:
        lr = 1e-6

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    writer.add_scalar("train/loss", loss.item(), step)

    if step % 100 == 0:
        net.eval()
        y = net(x)
        loss = criterion(y, gt)
        writer.add_scalar("val/loss", loss.item(), step)
        for i in range(32):
            G = y[i].cpu().data.numpy()
            GT = gt[i].cpu().data.numpy()
            writer.add_image("train/gen{}.png".format(i), G, step)
            writer.add_image("train/ground_truth{}.png".format(i), GT, step)

    if step % 1000 == 0:
        save_model()
    step += 1
