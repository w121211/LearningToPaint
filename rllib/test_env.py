"""Example of a custom gym environment and model. Run this for a demo.

This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search

You can visualize experiment results in ~/ray_results using TensorBoard.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork

# from gym.spaces import Discrete, Box
from gym import spaces

import ray
from ray import tune
from ray.rllib.utils import try_import_tf
from ray.tune import grid_search

tf = try_import_tf()


class SimpleCorridor(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.

    You can configure the length of the corridor via the env config."""

    def __init__(self, config):
        self.end_pos = config["corridor_length"]
        self.cur_pos = 0
        self.action_space = spaces.Discrete(2)
        # self.observation_space = Box(0.0, self.end_pos, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "a": spaces.Box(0.0, self.end_pos, shape=(1,), dtype=np.float32),
                "b": spaces.Box(0.0, self.end_pos, shape=(1,), dtype=np.float32),
            }
        )

    def reset(self):
        self.cur_pos = 0
        # return [self.cur_pos]
        return {"a": [self.cur_pos], "b": [self.cur_pos]}

    def step(self, action):
        """
        Returns:
            obs
            reward
            done
            log
        """
        assert action in [0, 1], action
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        elif action == 1:
            self.cur_pos += 1
        done = self.cur_pos >= self.end_pos
        return {"a": [self.cur_pos], "b": [self.cur_pos]}, 1 if done else 0, done, {}


class CustomModel(TFModelV2):
    """Example of a custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CustomModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        x1 = tf.keras.Input(shape=(1,))
        x2 = tf.keras.Input(shape=(1,))
        x = tf.keras.layers.concatenate([x1, x2], -1)
        x = tf.keras.layers.Dense(16, activation="relu")(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)

        layer_out = tf.keras.layers.Dense(num_outputs, activation=None)(x)
        value_out = tf.keras.layers.Dense(1, activation=None)(x)

        self.model = tf.keras.Model([x1, x2], [layer_out, value_out])
        self.register_variables(self.model.variables)

    def forward(self, input_dict, state, seq_lens):
        print(input_dict["obs"]["a"])
        model_out, self._value_out = self.model(
            [input_dict["obs"]["a"], input_dict["obs"]["b"]]
        )
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


if __name__ == "__main__":
    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    ray.init()
    ModelCatalog.register_custom_model("my_model", CustomModel)
    tune.run(
        "PPO",
        stop={"timesteps_total": 10000},
        config={
            # "log_level": "ERROR",
            "eager": False,
            "env": SimpleCorridor,  # or "corridor" if registered above
            "model": {"custom_model": "my_model"},
            "vf_share_layers": True,
            "lr": grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
            # "lr": grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
            "num_workers": 1,  # parallelism
            "env_config": {"corridor_length": 5},
        },
    )
