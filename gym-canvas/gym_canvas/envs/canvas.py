import os, subprocess, time, signal

import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms

import gym
from gym import error, spaces
from gym.utils import seeding, EzPickle


class CanvasEnv(gym.Env, EzPickle):
    def __init__(self):
        self.max_step = 5
        self.cur_step = 0
        self.width = 64
        self.obj_w = 20
        self.obj_x = None
        self.obj_y = None
        self.num_obj = 1

        # avoid using spaces.Tuple <- not supported by pytorch-a2c-ppo
        # merge all inputs together: (cur_image, target_image, object_infos)
        self.obs_shapes = (
            (1 * self.width * self.width * 2),  # (channel, width, height, num_images)
            (self.num_obj * 4),  # (num_obj, num_attributes)
        )
        self.observation_space = spaces.Box(
            low=-10000,
            high=10000,
            shape=(1 * self.width * self.width * 2 + self.num_obj * 4,),
            dtype=np.float32,
        )
        # self.action_space = spaces.Box(low=0, high=63, shape=(2,), dtype=np.uint8)
        self.action_space = spaces.Tuple(
            (spaces.Discrete(1), spaces.Box(low=0, high=self.width, shape=(2,)))
        )
        self.viewer = None
        self.cur_im = None
        self.target_im = None
        self.field_info = None

    def _render(self, coord):
        x_0, y_0 = coord
        transform = transforms.ToTensor()
        im = Image.new("L", (self.width, self.width))
        draw = ImageDraw.Draw(im)
        draw.rectangle([x_0, y_0, x_0 + self.obj_w, y_0 + self.obj_w], fill=255)
        return transform(im)

    def _obs(self):
        return torch.cat(
            (self.target_im.view(-1), self.cur_im.view(-1), self.field_info.view(-1)),
            dim=0,
        )

    def step(self, action):
        # print(action)
        x0, y0 = action
        self.field_info = torch.FloatTensor(
            ((x0, y0, x0 + self.obj_w, y0 + self.obj_w))
        )
        self.cur_im = self._render((x0, y0))
        reward = -((x0 - self.obj_x) ** 2) + -((y0 - self.obj_y) ** 2)
        done = self.cur_step > self.max_step
        return self._obs(), reward, done, {"episode": {"r": reward}}

    def reset(self):
        x0, y0 = 0, 0  # initial obj coord
        self.obj_x, self.obj_y = np.random.randint(self.width - self.obj_w, size=2)
        self.target_im = self._render((self.obj_x, self.obj_y))
        self.cur_im = self._render((x0, y0))
        self.field_info = torch.FloatTensor(
            ((x0, y0, x0 + self.obj_w, y0 + self.obj_w))
        )
        return self._obs()

    def render(self, mode="human", close=False):
        pass

    def close(self):
        pass

