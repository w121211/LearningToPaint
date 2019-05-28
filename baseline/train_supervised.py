#!/usr/bin/env python3
import cv2
import random
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from env import Paint
from DRL.actor import ResNet

# from DRL.ddpg import
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CoordConv implementation: https://arxiv.org/abs/1807.03247
coord = torch.zeros([1, 2, 128, 128])
for i in range(128):
    for j in range(128):
        coord[0, 0, i, j] = i / 127.0
        coord[0, 1, i, j] = j / 127.0
coord = coord.to(device)

# settings
action_dim = 4
n_frames_per_step = 1


def train(batch_size, max_episode_length=10):
    env = Paint(batch_size, max_episode_length)
    actor = ResNet(
        9, 18, (action_dim + 3) * n_frames_per_step
    )  # target, canvas, stepnum, coordconv 3 + 3 + 1 + 2
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(actor.parameters(), lr=1e-2)

    for step in range(50000):
        state, y_target = env.reset_with_gen()
        y_target = y_target.view(batch_size, -1)
        state = torch.cat(
            (
                state[:, :6].float() / 255,
                state[:, 6:7].float() / max_episode_length,
                coord.expand(state.shape[0], 2, 128, 128),
            ),
            1,
        )
        actor.zero_grad()
        y = actor(state)
        loss = loss_fn(y, y_target)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print("step %d: loss %f" % (step, loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int, help="minibatch size")
    parser.add_argument("--episode_steps", default=1, type=int)
    parser.add_argument(
        "--train_steps", default=100000, type=int, help="total traintimes"
    )
    parser.add_argument(
        "--resume", default=None, type=str, help="Resuming model path for testing"
    )
    parser.add_argument(
        "--output", default="./model", type=str, help="Resuming model path for testing"
    )
    parser.add_argument("--seed", default=1234, type=int, help="random seed")
    args = parser.parse_args()

    # args.output = get_output_folder(args.output, "Paint")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    train(args.batch_size, args.episode_steps)
