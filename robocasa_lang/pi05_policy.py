"""π0.5 (openpi websocket server) behind the same interface run_lang_eval.py uses for the Xiaomi-Robotics-1 upstream.

The observation packing copies robocasa-benchmark/openpi examples/robocasa/main.py @ 5a6beda: three cameras resized
with padding to 224, state = eef pos + eef rot (relative) + base pos + base rot + gripper qpos, the instruction under
"prompt", one query per action chunk. No observation history (π0.5 sees the latest frame only), so sample_history
returns the newest entry.
"""
import numpy as np

CAMERA_KEYS = ("video.robot0_agentview_left", "video.robot0_eye_in_hand", "video.robot0_agentview_right")
STATE_KEYS = ("state.end_effector_position_relative", "state.end_effector_rotation_relative", "state.base_position",
              "state.base_rotation", "state.gripper_qpos")
RESIZE = 224
REPLAN_STEPS = 5  # upstream default


def collect_images(obs):
    return {k: np.ascontiguousarray(obs[k]) for k in CAMERA_KEYS}


def observation_to_state(obs):
    return np.concatenate([obs[k] for k in STATE_KEYS], axis=0)


def sample_history(q, n, interval):
    return q[-1]


def make_video_frame(obs):
    return np.concatenate([np.asarray(obs[CAMERA_KEYS[0]]), np.asarray(obs[CAMERA_KEYS[2]])], axis=1)


class Client:
    def __init__(self, host="127.0.0.1", port=8000, resize=RESIZE):
        from openpi_client import image_tools
        from openpi_client import websocket_client_policy
        self._tools, self.resize = image_tools, resize
        self._ws = websocket_client_policy.WebsocketClientPolicy(host, port)

    def _img(self, im):
        return self._tools.convert_to_uint8(self._tools.resize_with_pad(im, self.resize, self.resize))

    def infer(self, state, images, instruction):
        element = {"observation/image": self._img(images[CAMERA_KEYS[0]]),
                   "observation/wrist_image": self._img(images[CAMERA_KEYS[1]]),
                   "observation/right_image": self._img(images[CAMERA_KEYS[2]]),
                   "observation/state": state, "prompt": instruction}
        return list(self._ws.infer(element)["actions"])

    def close(self):
        pass
