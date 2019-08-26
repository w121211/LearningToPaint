import os, subprocess, time, signal

import numpy as np
from PIL import Image, ImageDraw

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
# from gym.utils import seeding, EzPickle

tf = try_import_tf()


class CanvasEnv(gym.Env):
    def __init__(self, config):
        self.max_step = 5
        self.cur_step = 0
        self.width = 64
        self.obj_w = 20
        self.obj_x = None
        self.obj_y = None
        self.num_obj = 1

        self.observation_space = spaces.Dict(
            {
                "target_im": spaces.Box(
                    low=0, high=1, shape=(1, self.width, self.width)
                ),  # (H, W, C)
                "cur_im": spaces.Box(
                    low=0, high=1, shape=(1, self.width, self.width)
                ),  # (H, W, C)
                "obj_status": spaces.Box(low=0, high=1, shape=(self.num_obj * 4,)),
            }
        )
        self.action_space = spaces.Tuple([
                spaces.Discrete(1), 
                spaces.Box(low=0, high=self.width, shape=(2,)),])
        self.cur_im = None
        self.target_im = None
        self.obj_status = None
        self.viewer = None

    def _render(self, coord):
        x_0, y_0 = coord
        # transform = transforms.ToTensor()
        im = Image.new("L", (self.width, self.width))
        draw = ImageDraw.Draw(im)
        draw.rectangle([x_0, y_0, x_0 + self.obj_w, y_0 + self.obj_w], fill=255)
        # return transform(im)
        return np.array(im)

    def _obs(self):
        return {
            "target_im": self.target_im,
            "cur_im": self.cur_im,
            "obj_status": self.obj_status, 
        }

    def step(self, action):
        """
        Args:
            action: spaces.Tuple(spaces.Discrete, spaces.Box), 
        Return:
            obs: target_im (H, W, C), cur_im (H, W, C), field_info (x0, y0)
        """
        # print(action)
        x0, y0 = action

        # self.field_info = torch.FloatTensor(
        #     ((x0, y0, x0 + self.obj_w, y0 + self.obj_w))
        # )
        self.obj_status = np.array([x0, y0, x0 + self.obj_w, y0 + self.obj_w], dtype=np.float32)
        self.cur_im = self._render((x0, y0))
        reward = -((x0 - self.obj_x) ** 2) + -((y0 - self.obj_y) ** 2)
        done = self.cur_step > self.max_step
        return self._obs(), reward, done, {"episode": {"r": reward}}

    def reset(self):
        x0, y0 = 0, 0  # initial obj coord
        self.obj_x, self.obj_y = np.random.randint(self.width - self.obj_w, size=2)
        self.target_im = self._render((self.obj_x, self.obj_y))
        self.cur_im = self._render((x0, y0))
        self.obj_status = np.array([x0, y0, x0 + self.obj_w, y0 + self.obj_w], dtype=np.float32)
        return self._obs()

    def render(self, mode="human", close=False):
        pass

    def close(self):
        pass

# class MyConvNet(TFModelV2):
#     def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#         super(MyConvNet, self).__init__(
#             obs_space, action_space, num_outputs, model_config, name
#         )

#         activation = get_activation_fn(model_config.get("conv_activation"))
#         filters = model_config.get("conv_filters")
#         if not filters:
#             filters = _get_filter_config(obs_space.shape)
#         no_final_linear = model_config.get("no_final_linear")
#         vf_share_layers = model_config.get("vf_share_layers")

#         inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
#         last_layer = inputs

#         # Build the action layers
#         for i, (out_size, kernel, stride) in enumerate(filters[:-1], 1):
#             last_layer = tf.keras.layers.Conv2D(
#                 out_size,
#                 kernel,
#                 strides=(stride, stride),
#                 activation=activation,
#                 padding="same",
#                 name="conv{}".format(i),
#             )(last_layer)
#         out_size, kernel, stride = filters[-1]
#         if no_final_linear:
#             # the last layer is adjusted to be of size num_outputs
#             last_layer = tf.keras.layers.Conv2D(
#                 num_outputs,
#                 kernel,
#                 strides=(stride, stride),
#                 activation=activation,
#                 padding="valid",
#                 name="conv_out",
#             )(last_layer)
#             conv_out = last_layer
#         else:
#             last_layer = tf.keras.layers.Conv2D(
#                 out_size,
#                 kernel,
#                 strides=(stride, stride),
#                 activation=activation,
#                 padding="valid",
#                 name="conv{}".format(i + 1),
#             )(last_layer)
#             conv_out = tf.keras.layers.Conv2D(
#                 num_outputs, [1, 1], activation=None, padding="same", name="conv_out"
#             )(last_layer)

#         # Build the value layers
#         if vf_share_layers:
#             last_layer = tf.squeeze(last_layer, axis=[1, 2])
#             value_out = tf.keras.layers.Dense(
#                 1,
#                 name="value_out",
#                 activation=None,
#                 kernel_initializer=normc_initializer(0.01),
#             )(last_layer)
#         else:
#             # build a parallel set of hidden layers for the value net
#             last_layer = inputs
#             for i, (out_size, kernel, stride) in enumerate(filters[:-1], 1):
#                 last_layer = tf.keras.layers.Conv2D(
#                     out_size,
#                     kernel,
#                     strides=(stride, stride),
#                     activation=activation,
#                     padding="same",
#                     name="conv_value_{}".format(i),
#                 )(last_layer)
#             out_size, kernel, stride = filters[-1]
#             last_layer = tf.keras.layers.Conv2D(
#                 out_size,
#                 kernel,
#                 strides=(stride, stride),
#                 activation=activation,
#                 padding="valid",
#                 name="conv_value_{}".format(i + 1),
#             )(last_layer)
#             last_layer = tf.keras.layers.Conv2D(
#                 1, [1, 1], activation=None, padding="same", name="conv_value_out"
#             )(last_layer)
#             value_out = tf.squeeze(last_layer, axis=[1, 2])

#         self.base_model = tf.keras.Model(inputs, [conv_out, value_out])
#         self.register_variables(self.base_model.variables)

#     def forward(self, input_dict, state, seq_lens):
#         # explicit cast to float32 needed in eager
#         model_out, self._value_out = self.base_model(
#             tf.cast(input_dict["obs"], tf.float32)
#         )
#         return tf.squeeze(model_out, axis=[1, 2]), state

#     def value_function(self):
#         return tf.reshape(self._value_out, [-1])


class CustomModel(TFModelV2):
    """Example of a custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)
        self.model = FullyConnectedNetwork(obs_space, action_space,
                                           num_outputs, model_config, name)
        self.register_variables(self.model.variables())

    def forward(self, input_dict, state, seq_lens):
        print(input_dict)
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()


if __name__ == "__main__":
    # Can also register the env creator function explicitly with:
    # register_env("canvas", lambda config: CanvasEnv(config))
    ray.init()
    ModelCatalog.register_custom_model("my_model", CustomModel)
    tune.run(
        "PPO",
        stop={"timesteps_total": 10000},
        config={
            "env": CanvasEnv,  # or "corridor" if registered above
            "model": {"custom_model": "my_model"},
            "vf_share_layers": True,
            # "lr": grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
            "num_workers": 1,  # parallelism
            # "env_config": {"corridor_length": 5},
        },
    )

x = {'obs': OrderedDict([('cur_im', <tf.Tensor 'default_policy/Reshape:0' shape=(?, 1, 64, 64) dtype=float32>), ('obj_status', <tf.Tensor 'default_policy/Reshape_1:0' shape=(?, 4) dtype=float32>), ('target_im', <tf.Tensor 'default_policy/Reshape_2:0' shape=(?, 1, 64, 64) dtype=float32>)]), 'prev_actions': <tf.Tensor 'default_policy/action:0' shape=(?, 3) dtype=float32>, 'prev_rewards': <tf.Tensor 'default_policy/prev_reward:0' shape=(?,) dtype=float32>, 'is_training': <tf.Tensor 'default_policy/PlaceholderWithDefault:0' shape=() dtype=bool>, 'obs_flat': <tf.Tensor 'default_policy/observation:0' shape=(?, 8196) dtype=float32>}