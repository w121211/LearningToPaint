import os, subprocess, time, signal

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import ray
from ray import tune
from ray.rllib.utils import try_import_tf
from ray.tune import grid_search
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
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

class SlimConv2d(nn.Module):
    """Simple mock of tf.slim Conv2d"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel,
                 stride=1,
                 padding=0,
                 initializer=nn.init.xavier_uniform_,
                 activation_fn=nn.ReLU,
                 bias_init=0):
        super(SlimConv2d, self).__init__()
        layers = []
        # if padding:
        #     layers.append(nn.ZeroPad2d(padding))
        conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding)
        if initializer:
            initializer(conv.weight)
        nn.init.constant_(conv.bias, bias_init)

        layers.append(conv)
        if activation_fn:
            layers.append(activation_fn())
        self._model = nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)


class SlimFC(nn.Module):
    """Simple PyTorch version of `linear` function"""

    def __init__(self,
                 in_size,
                 out_size,
                 initializer=None,
                 activation_fn=None,
                 bias_init=0):
        super(SlimFC, self).__init__()
        layers = []
        linear = nn.Linear(in_size, out_size)
        if initializer:
            initializer(linear.weight)
        nn.init.constant_(linear.bias, bias_init)
        layers.append(linear)
        if activation_fn:
            layers.append(activation_fn())
        self._model = nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)


class VisionNetwork(TorchModelV2, nn.Module):
    """Generic vision network."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self._convs = nn.Sequential(
            SlimConv2d(2, 8, 3, stride, padding=1),
            SlimConv2d(8, 16, 3, stride, padding=1),
            SlimConv2d(16, 16, 3, stride, padding=1),
            SlimConv2d(16, 16, 3, stride, padding=1),
        )

        self._logits = SlimFC(
            out_channels, num_outputs, initializer=nn.init.xavier_uniform_)
        self._value_branch = SlimFC(
            out_channels, 1, initializer=normc_initializer())
        self._cur_value = None

    def forward(self, input_dict, state, seq_lens):
        features = self._hidden_layers(input_dict["obs"].float())
        logits = self._logits(features)
        self._cur_value = self._value_branch(features).squeeze(1)
        return logits, state

    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

    def _hidden_layers(self, obs):
        # res = self._convs(obs.permute(0, 3, 1, 2))  # switch to channel-major
        res = res.squeeze(3)
        res = res.squeeze(2)
        return res

if __name__ == "__main__":
    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    ray.init()
    ModelCatalog.register_custom_model("my_model", CustomModel)
    tune.run(
        "PPO",
        stop={
            "timesteps_total": 10000,
        },
        config={
            "env": SimpleCorridor,  # or "corridor" if registered above
            "model": {
                "custom_model": "my_model",
            },
            "vf_share_layers": True,
            "lr": grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
            "num_workers": 1,  # parallelism
            "env_config": {
                "corridor_length": 5,
            },
        },
    )