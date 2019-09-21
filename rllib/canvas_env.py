import os, subprocess, time, signal

import numpy as np
from PIL import Image, ImageDraw

import ray
from ray import tune
from ray.rllib.utils import try_import_tf
from ray.tune import grid_search
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
import gym
from gym import error, spaces

tf = try_import_tf()

CANVAS_WIDTH = 9
OBJ_WIDTH = 3


class CanvasEnv(gym.Env):
    def __init__(self, config):
        self.max_step = 5
        self.cur_step = 0
        self.width = CANVAS_WIDTH
        self.obj_w = OBJ_WIDTH
        self.obj_x = None
        self.obj_y = None
        self.num_obj = 1

        self.observation_space = spaces.Dict(
            {
                "target_im": spaces.Box(
                    low=0, high=1, shape=(self.width, self.width, 1)
                ),  # (H, W, C)
                "cur_im": spaces.Box(
                    low=0, high=1, shape=(self.width, self.width, 1)
                ),  # (H, W, C)
                "obj_status": spaces.Box(
                    low=-10, high=10, shape=(self.num_obj, self.num_obj * 4)
                ),
            }
        )
        self.action_space = spaces.Tuple(
            [spaces.Discrete(1), spaces.Box(low=0, high=self.width, shape=(2,))]
        )
        self.cur_im = None
        self.target_im = None
        self.obj_status = None
        self.viewer = None
        self.target_coords = None

    def _render(self, x0, y0):
        im = Image.new("L", (self.width, self.width))
        draw = ImageDraw.Draw(im)
        draw.rectangle([x0, y0, x0 + self.obj_w, y0 + self.obj_w], fill=255)

        x = np.array(im, dtype=np.float32) / 255.0  # normalize
        x = np.expand_dims(x, axis=-1)  # (H, W, C=1)
        return x

    def _obs(self):
        return {
            "target_im": self.target_im,
            "cur_im": self.cur_im,
            "obj_status": self.obj_status,
        }

    def step(self, action):
        """
        Args:
            action: list[int, np.array]
        Return:
            obs: target_im (H, W, C), cur_im (H, W, C), field_info (x0, y0)
        """
        obj_id, coord = action
        coord *= self.width
        x0, y0 = coord
        self.obj_status = np.array(
            [[x0, y0, (x0 + self.obj_w), (y0 + self.obj_w)]], dtype=np.float32
        ) / self.width
        self.cur_im = self._render(x0, y0)
        reward = -((x0 - self.target_coords[0, 0]) ** 2) + -(
            (y0 - self.target_coords[0, 1]) ** 2
        )
        done = self.cur_step > self.max_step
        # return self._obs(), reward, done, {"episode": {"r": reward}}
        return self._obs(), 1, done, {"episode": {"r": reward}}

    def reset(self):
        obj_coords = []
        for _ in range(self.num_obj):
            _x0, _y0 = np.random.randint(self.width - self.obj_w, size=2)
            obj_coords.append([_x0, _y0])
        self.target_coords = np.array(obj_coords, dtype=np.float32)

        obj_coords = []
        for _ in range(self.num_obj):
            _x0, _y0 = 0, 0
            obj_coords.append([_x0, _y0, (_x0 + self.obj_w), (_y0 + self.obj_w)])
        self.obj_status = np.array(obj_coords, dtype=np.float32) / self.width

        self.target_im = self._render(*(self.target_coords[0]))
        self.cur_im = self._render(0, 0)

        return self._obs()

    def render(self, mode="human", close=False):
        pass

    def close(self):
        pass


class MyModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        input_img = tf.keras.Input(shape=(CANVAS_WIDTH, CANVAS_WIDTH, 1))  # (H, W, C)
        x = tf.keras.layers.Conv2D(8, 3, activation="relu", padding="same")(input_img)
        x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
        x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
        x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
        x = tf.keras.layers.Conv2D(1, 3, padding="same")(x)
        out = tf.keras.layers.Flatten()(x)

        cnn_model = tf.keras.Model(input_img, out)

        cur_im = tf.keras.Input(shape=(CANVAS_WIDTH, CANVAS_WIDTH, 1))
        target_im = tf.keras.Input(shape=(CANVAS_WIDTH, CANVAS_WIDTH, 1))

        # The vision model will be shared, weights and all
        out_cur = cnn_model(cur_im)
        out_target = cnn_model(target_im)
        obj_status = tf.keras.Input(shape=(1, 4))  # (num_obj, 4=(x0, y0, x1, y1))

        x = tf.keras.layers.Flatten()(obj_status)
        x = tf.keras.layers.concatenate([out_cur, out_target, x])
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)

        layer_out = tf.keras.layers.Dense(num_outputs, activation=None)(x)
        value_out = tf.keras.layers.Dense(1, activation=None)(x)

        self.model = tf.keras.Model(
            [cur_im, target_im, obj_status], [layer_out, value_out]
        )
        self.register_variables(self.model.variables)

    def forward(self, input_dict, state, seq_lens):
        print(input_dict["obs"]["cur_im"])
        model_out, self._value_out = self.model(
            [
                input_dict["obs"]["cur_im"],
                input_dict["obs"]["target_im"],
                input_dict["obs"]["obj_status"],
            ]
        )
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


if __name__ == "__main__":
    # Can also register the env creator function explicitly with:
    # register_env("canvas", lambda config: CanvasEnv(config))
    ray.init()
    ModelCatalog.register_custom_model("my_model", MyModel)
    tune.run(
        "PPO",
        stop={"timesteps_total": 10000},
        config={
            "log_level": "INFO",
            "num_workers": 1,  # parallelism
            "num_gpus": 0,
            "env": CanvasEnv,  # or "corridor" if registered above
            # "env_config": {"corridor_length": 5},
            "model": {"custom_model": "my_model"},
            # "model_config": {},
            # "lr": grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
        },
    )
