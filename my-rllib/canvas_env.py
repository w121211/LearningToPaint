# %%writefile /content/LearningToPaint/rllib/canvas_env.py
import os, subprocess, time, signal

import numpy as np
from PIL import Image, ImageDraw

import gym
from gym import error, spaces
import ray
from ray import tune
from ray.rllib.utils import try_import_tf
from ray.tune import grid_search
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation.postprocessing import discount
from ray.rllib.policy.tf_policy_template import build_tf_policy

tf = try_import_tf()

CANVAS_WIDTH = 64
OBJ_WIDTH = 10
N_ITEMS = 2


class CanvasEnv(gym.Env):
    def __init__(self, config):
        self.max_step = 10
        self.cur_step = 0
        self.width = CANVAS_WIDTH
        self.obj_wh = np.array([OBJ_WIDTH, OBJ_WIDTH], dtype=np.float32) / self.width
        self.n_items = N_ITEMS

        self.observation_space = spaces.Dict(
            {
                "target_im": spaces.Box(
                    low=0, high=1, shape=(self.width, self.width, 1)
                ),  # (H, W, C)
                "cur_im": spaces.Box(
                    low=0, high=1, shape=(self.width, self.width, 1)
                ),  # (H, W, C)
                "cur_coord": spaces.Box(low=-10, high=10, shape=(self.n_items, 4)),
            }
        )
        self.action_space = spaces.Tuple(
            [
                spaces.Discrete(self.n_items),  # choosed item
                spaces.Discrete(5),  # dx
                spaces.Discrete(5),  # dy
                # spaces.Box(low=0, high=self.width, shape=(2,))
            ]
        )
        self.action_map = np.array([-8, 8, -1, 1, 0], dtype=np.float32) / self.width

        self.target_im = None
        self.target_coord = None  # (n_obj, 4=(x0, y0, x1, y1))
        self.cur_im = None
        self.cur_coord = None  # (n_obj, 4=(x0, y0, x1, y1))
        self.viewer = None

    def reset(self):
        self.cur_step = 0
        xy0 = np.random.rand(self.n_items, 2)
        self.target_coord = np.concatenate(
            [xy0, xy0 + np.tile(self.obj_wh, (self.n_items, 1))], axis=1
        )
        self.cur_coord = np.tile(
            np.array([0, 0, *tuple(self.obj_wh)], dtype=np.float32), (self.n_items, 1)
        )
        self.target_im = self._render(self.target_coord)
        self.cur_im = self._render(self.cur_coord)
        return self._obs()

    def step(self, action):
        """
        Args:
            action: [int, int]
        Return:
            obs: target_im (H, W, C), cur_im (H, W, C), field_info (x0, y0)
        """
        # print(self.action_map[action[0]], self.action_map[action[1]])
        idx = action[0]
        dmove = np.array(
            [self.action_map[action[1]], self.action_map[action[2]]], dtype=np.float32
        )
        xy0 = self.cur_coord[idx, :2] + dmove
        self.cur_coord[idx] = np.concatenate([xy0, xy0 + self.obj_wh], axis=0)
        self.cur_im = self._render(self.cur_coord)

        reward = self._reward(self.cur_coord, self.target_coord)

        self.cur_step += 1
        done = self.cur_step >= self.max_step

        return self._obs(), reward, done, {}

    def _render(self, coord: np.array):
        """
        Args: coord: (n_items, 4)
        """
        coord = (coord * self.width).astype(np.int16)
        im = Image.new("L", (self.width, self.width))
        draw = ImageDraw.Draw(im)
        for i, c in enumerate(coord):
            if i == 0:
                draw.rectangle(tuple(c), fill=255)
            else:
                draw.ellipse(tuple(c), fill=255)
        x = np.array(im, dtype=np.float32) / 255.0  # normalize
        x = np.expand_dims(x, axis=-1)  # (H, W, C=1)
        return x

    def _obs(self):
        return {
            "target_im": self.target_im,
            "cur_im": self.cur_im,
            "cur_coord": self.cur_coord,
        }

    def _reward(self, xy_a: np.array, xy_b: np.array):
        dist = np.linalg.norm(xy_a - xy_b, axis=1)
        r = -1 * dist / 2 + 1
        r = np.clip(r, -1, None)
        r = np.sum(r)
        #     elif r > 0:
        #         r *= 0.05 ** self.cur_step  # 衰退因子
        return r

    def _denorm(self, a: np.array):
        return (a * self.width).astype(np.int16)

    def _regression_step(self, action):
        """@Deprecated
        Args:
            action: list[obj_id: int, coord: np.array]
        Return:
            obs: target_im (H, W, C), cur_im (H, W, C), field_info (x0, y0)
        """
        obj_id, coord = action
        coord *= self.width
        x0, y0 = coord
        self.obj_status = (
            np.array([[x0, y0, (x0 + self.obj_w), (y0 + self.obj_w)]], dtype=np.float32)
            / self.width
        )
        self.cur_im = self._render(x0, y0)
        reward = -(
            ((x0 - self.target_coords[0, 0]) ** 2)
            + ((y0 - self.target_coords[0, 1]) ** 2)
        )
        self.cur_step += 1
        done = self.cur_step >= self.max_step
        return self._obs(), reward, done, {"episode": {"r": reward}}

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
        cur_coord = tf.keras.Input(shape=(N_ITEMS, 4))  # (n_items, 4=(x0, y0, x1, y1))

        x = tf.keras.layers.Flatten()(cur_coord)
        x = tf.keras.layers.concatenate([out_cur, out_target, x])
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)

        layer_out = tf.keras.layers.Dense(num_outputs, activation=None)(x)
        value_out = tf.keras.layers.Dense(1, activation=None)(x)

        self.model = tf.keras.Model(
            [cur_im, target_im, cur_coord], [layer_out, value_out]
        )
        self.register_variables(self.model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.model(
            [
                input_dict["obs"]["cur_im"],
                input_dict["obs"]["target_im"],
                input_dict["obs"]["cur_coord"],
            ]
        )
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


def policy_gradient_loss(policy, model, dist_class, train_batch):
    # print(train_batch)
    logits, _ = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)
    return -tf.reduce_mean(
        action_dist.logp(train_batch["actions"]) * train_batch["returns"]
    )


def calculate_advantages(policy, sample_batch, other_agent_batches=None, episode=None):
    # print(sample_batch)
    sample_batch["returns"] = discount(sample_batch["rewards"], 0.99)
    return sample_batch


MyTFPolicy = build_tf_policy(
    name="MyTFPolicy", loss_fn=policy_gradient_loss, postprocess_fn=calculate_advantages
)

MyTrainer = build_trainer(name="MyCustomTrainer", default_policy=MyTFPolicy)


if __name__ == "__main__":
    ray.init()

    ModelCatalog.register_custom_model("my_model", MyModel)
    ray.tune.registry.register_env("canvas", lambda config: CanvasEnv(config))

    tune.run(
        "PPO",
        # MyTrainer,
        checkpoint_at_end=True,
        stop={"timesteps_total": 1000000},
        config={
            # "log_level": "INFO",
            "log_sys_usage": False,
            "num_workers": 7,  # parallelism
            "num_gpus": 1,
            "env": CanvasEnv,  # or "corridor" if registered above
            # "env_config": {"corridor_length": 5},
            "model": {"custom_model": "my_model"},
            # "model_config": {},
            # "lr": grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
            # "vf_share_layers": True,
            # "train_batch_size": 2000,
        },
    )

    # import pickle
    # from ray.rllib.agents import ppo

    # with open("/root/ray_results/PPO/PPO_CanvasEnv_0_2019-10-03_15-47-36_3odsslh/params.pkl", "rb") as f:
    #     config = pickle.load(f)
    #     print(config)

    # agent = ppo.PPOTrainer(env="canvas", config=config)
    # agent.restore("/root/ray_results/PPO/PPO_CanvasEnv_0_2019-10-03_15-47-36_3odsslh/checkpoint_5/checkpoint-5")

    # print(agent.workers)

    # from ray.rllib import rollout
    # parser = rollout.create_parser()
    # args = parser.parse_args()
    # args.env = "canvas"
    # args.checkpoint = ""
    # rollout.run(args, parser)
