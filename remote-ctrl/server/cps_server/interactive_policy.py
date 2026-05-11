"""
Runs the policy in an interactive manner.

ESC+BACKSPACE = PANIC: kill program

SPACE BAR while idle: go to laying down position

SPACE BAR while laying down: go to upward position

otherwise: control direction with arrow keys.
           no direction means to just idle in place.


"""
from abc import ABC, abstractmethod

from math import pi
import numpy as np
import time
from numpy.linalg import norm
import traceback
from datetime import datetime
import pathlib
import json
import keyboard

zero_shift_dics = {
    "BR_INNER_SHOULDER": 0.0,
    "BR_OUTER_SHOULDER": -0.308,
    "BR_ELBOW": -0.211,
    "FR_INNER_SHOULDER": 0.0,
    "FR_OUTER_SHOULDER": -0.308,
    "FR_ELBOW": -0.231,
    "FL_INNER_SHOULDER": 0.0,
    "FL_OUTER_SHOULDER": -0.298,
    "FL_ELBOW": -0.191,
    "BL_INNER_SHOULDER": 0.0,
    "BL_OUTER_SHOULDER": -0.308,
    "BL_ELBOW": -0.231,
    "NO_KEY": 0.0,
}

# FOR THE NEW SPIDER
zero_shift_dics["BR_OUTER_SHOULDER"] = -0.95
zero_shift_dics["BR_ELBOW"] = 2.35
zero_shift_dics["FR_OUTER_SHOULDER"] = -0.95
zero_shift_dics["FR_ELBOW"] = 2.35
zero_shift_dics["FL_OUTER_SHOULDER"] = -0.95
zero_shift_dics["FL_ELBOW"] = 2.35
zero_shift_dics["BL_OUTER_SHOULDER"] = -0.95
zero_shift_dics["BL_ELBOW"] = 2.35

POSITIVE_JOINTS = {
    "BR_OUTER_SHOULDER",
    "FR_OUTER_SHOULDER",
    "FL_OUTER_SHOULDER",
    "BL_OUTER_SHOULDER",
}

FIXED_ANGLE = None
ACTION_DELTA = None
CONSTANT_ELBOW = None

#FIXED_ANGLE = np.pi * ((90.0 - 7.0) / 180.0)
#ACTION_DELTA = 0.2
#CONSTANT_ELBOW = 2.50

N_STEPS = 32

def dnx_to_mujoco(angle, motor_key):
    if motor_key in POSITIVE_JOINTS:
        # front_right_elbow and back_left_elbow are the only motors that don't have flipped sign
        return float((angle-2048)*pi*0.087891/180 + zero_shift_dics[motor_key])
    else:
        return float((2048-angle)*pi*0.087891/180 + zero_shift_dics[motor_key])

def mujoco_to_dnx(angle, motor_key):
    if motor_key in POSITIVE_JOINTS:
        # front_right_elbow and back_left_elbow are the only motors that don't have flipped sign
        v = int(2048 + round(180*(angle-zero_shift_dics[motor_key])/(pi*0.087891)))
    else:
        v = int(2048 - round(180*(angle-zero_shift_dics[motor_key])/(pi*0.087891)))

    while v < 0:
        v += 4096
    while v >= 4096:
        v -= 4096

    return v

def dnxvel_to_mujoco(vel, motor_key):
    if motor_key in POSITIVE_JOINTS:
        # front_right_elbow and back_left_elbow are the only motors that don't have flipped sign
        return vel*6*0.229*pi/180
        # 1 mujoco unit = 0.229 rpm = 0.229*360 deg per minute = 0.229*360/60 = 0.229*6 deg/s = 0.229*6*pi/180 rad/s
    else:
        return -vel*6*0.229*pi/180
        # 1 mujoco unit = 0.229 rpm = 0.229*360 deg per minute = 0.229*360/60 = 0.229*6 deg/s = 0.229*6*pi/180 rad/s



class SensorReader(ABC):
    @property
    @abstractmethod
    def output_size(self):
        """Returns the size of the output array for this reader"""
        pass

    @abstractmethod
    def read(self, *args, **kwargs):
        """Returns an 1d array like object with the sensor readings"""
        pass

    @abstractmethod
    def step(self, dt):
        """Some readers needs to keep an internal state. This function is called
        once for each time-step in the environment, where `dt` is the size of the
        time-step."""
        pass

    @abstractmethod
    def reset(self):
        """Resets the sensor reader"""
        pass


class NoFilter(SensorReader):
    def __init__(self, data):
        self._data = data

    @property
    def output_size(self):
        return len(self._data)

    def read(self):
        return self._data.copy()

    def step(self, dt):
        pass

    def reset(self):
        pass


class AccelerometerFilter(SensorReader):
    def __init__(self, data):
        self._data = data

    @property
    def output_size(self):
        return len(self._data)

    def read(self):
        # transform to robot body coordinate
        return np.array([self._data[1], self._data[0], self._data[2]]) / 9.81

    def step(self, dt):
        pass

    def reset(self):
        pass


class GyroscopeFilter(SensorReader):
    def _transform_gyro_data(data):
        # transform to robot body coordinate
        return np.array([-data[1], -data[0], -data[2]])

    def _Rx(a):
        return np.array([
            [1, 0, 0],
            [0, np.cos(np.radians(a)), -np.sin(np.radians(a))],
            [0, np.sin(np.radians(a)), np.cos(np.radians(a))]
        ])

    def _Ry(a):
        return np.array([
            [np.cos(np.radians(a)), 0, np.sin(np.radians(a))],
            [0, 1, 0],
            [-np.sin(np.radians(a)), 0, np.cos(np.radians(a))]
        ])

    def _Rz(a):
        return np.array([
            [np.cos(np.radians(a)), -np.sin(np.radians(a)), 0],
            [np.sin(np.radians(a)), np.cos(np.radians(a)), 0],
            [0, 0, 1]
        ])

    # Higher s means more adapted to accelerometer data
    def _interpolate_orientation(Rg, a, s):
        aw = Rg@a
        cross = np.cross(aw,np.array([0,0,-1]))
        n = cross/norm(cross)
        theta = np.arcsin(norm(cross)/norm(aw))
        mat = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
        new_R = (np.eye(3) + np.sin(s*theta)*mat + (1-np.cos(s*theta))*mat@mat)@Rg
        return new_R

    def _IMU_sim(R, gyro, acc, dt):
        deltas = [val*dt for val in gyro]
        newR = (
            R
            @GyroscopeFilter._Rz(deltas[2])
            @GyroscopeFilter._Ry(deltas[1])
            @GyroscopeFilter._Rx(deltas[0])
        )
        newR = GyroscopeFilter._interpolate_orientation(newR, acc, 0.2)
        return newR

    _R = np.array([
            [0, -1, 0],
            [-1, 0, 0],
            [0, 0, -1]
    ], dtype=np.float64)

    def __init__(self, gyro_data, accel_data):
        self._accelerometer_filter = AccelerometerFilter(accel_data)
        self._gyro_data = gyro_data
        # Rotation matrix of the gyroscope. The assumption here is that the
        # simulation starts with the body level.
        self._R = GyroscopeFilter._R

    @property
    def output_size(self):
        return len(self._R.flatten())

    def step(self, dt):
        self._accelerometer_filter.step(dt)
        accel_data = self._accelerometer_filter.read()
        gyro_data = GyroscopeFilter._transform_gyro_data(self._gyro_data)
        self._R = GyroscopeFilter._IMU_sim(self._R, gyro_data, accel_data, dt)

    def read(self):
        return self._R.copy().flatten()

    def reset(self):
        self._R = GyroscopeFilter._R



def get_obs(ctrl : "SpiderController", state):
    # Gets an observation in the MuJoCo format
    src_accel = ctrl.read_accel()
    src_gyro = ctrl.read_gyro()
    spider_data = ctrl.read_all_servos_RAM()

    mj_accel = np.array([src_accel[1], src_accel[0], src_accel[2]]) * (9.81 / 2.0)
    mj_gyro = np.array([-src_gyro[1], -src_gyro[0], -src_gyro[2]])

    if state["gyro"] is None:
        state["gyro"] = GyroscopeFilter(mj_gyro, mj_accel)
    else:
        state["gyro"]._gyro_data = mj_gyro
        state["gyro"]._accelerometer_filter._data = mj_accel
        state["gyro"].step(dt=0.25)

    fl_gyro = state["gyro"].read()
    fl_accel = state["gyro"]._accelerometer_filter.read()
    assert fl_gyro.shape == (9,)
    assert fl_accel.shape == (3,)

    src_positions = spider_data["PRESENT_POSITION"]
    src_velocity = spider_data["PRESENT_VELOCITY"]

    SERVO_ORDER = ctrl.get_servos()

    mj_positions = np.zeros((12,))
    for i in range(12):
        mj_positions[i] = dnx_to_mujoco(src_positions[i], SERVO_ORDER[i])
        hwstat = spider_data["HARDWARE_ERROR_STATUS"][i]
        if hwstat != 0:
            add_info(state, f"Hardware error 0x{hwstat:02x} on servo {SERVO_ORDER[i]}")

    if state["velocity"] is None:
        state["velocity"] = np.zeros((12,))
        state["last_position"] = mj_positions
    else:
        state["velocity"] = (mj_positions - state["last_position"]) / 0.25
        state["last_position"] = mj_positions

    #mj_velocity = state["velocity"]
    mj_velocity = np.array([dnxvel_to_mujoco(src_velocity[i], SERVO_ORDER[i]) for i in range(12)])

    mj_servos = np.concatenate(tuple(zip(mj_positions,mj_velocity)))
    assert mj_servos.shape == (24,)

    #mujoco_obs = np.concatenate((fl_accel, fl_gyro, mj_servos))
    mujoco_obs = mj_servos

    add_to_trajectory(state, "observation", {"mujoco": mujoco_obs.tolist(), "raw": spider_data})

    return mujoco_obs

def apply_action(ctrl : "SpiderController", action, state, steady=True):
    # Applies an action to the spider robot

    mj_action = action
    assert mj_action.shape == (12,), f"got action {mj_action}"

    SERVO_ORDER = ctrl.get_servos()
    assert len(SERVO_ORDER) == 12

    raw_action = [mujoco_to_dnx(a, sv) for a, sv in zip(mj_action.tolist(), SERVO_ORDER)]
    assert len(raw_action) == 12

    add_to_trajectory(state, "action", {"mujoco": mj_action.tolist(), "raw": raw_action})

    if steady:
        ctrl.move_all_servos_steady(*raw_action)
    else:
        ctrl.move_all_servos(*raw_action)



IDX_SHIFTS = {
    "0": {
        "obs": {"front_left": 18, "front_right":  6, "back_left": 12,  "back_right":  0},
        "act": {"front_left":  9, "front_right":  3, "back_left":  6,  "back_right":  0},
    },
}
ROT_90 = [
    ("front_left",  "back_left"),
    ("front_right", "front_left"),
    ("back_left",   "back_right"),
    ("back_right",  "front_right"),
]
IDX_SHIFTS["90"] = {"obs": {}, "act": {}}
for (src, dst) in ROT_90:
    IDX_SHIFTS["90"]["obs"][dst] = IDX_SHIFTS["0"]["obs"][src]
    IDX_SHIFTS["90"]["act"][dst] = IDX_SHIFTS["0"]["act"][src]
IDX_SHIFTS["180"] = {"obs": {}, "act": {}}
for (src, dst) in ROT_90:
    IDX_SHIFTS["180"]["obs"][dst] = IDX_SHIFTS["90"]["obs"][src]
    IDX_SHIFTS["180"]["act"][dst] = IDX_SHIFTS["90"]["act"][src]
IDX_SHIFTS["270"] = {"obs": {}, "act": {}}
for (src, dst) in ROT_90:
    IDX_SHIFTS["270"]["obs"][dst] = IDX_SHIFTS["180"]["obs"][src]
    IDX_SHIFTS["270"]["act"][dst] = IDX_SHIFTS["180"]["act"][src]

def step(ctrl, state, model, rot="0"):
    """
    Take a step.
    """
    if state["last_time"] is None:
        state["last_time"] = time.time()

    t_start = time.time()
    obs = get_obs(ctrl, state)

    (r_obs, r_act) = (IDX_SHIFTS["0"]["obs"], IDX_SHIFTS["0"]["act"])
    (s_obs, s_act) = (IDX_SHIFTS[rot]["obs"], IDX_SHIFTS[rot]["act"])

    # Rotate the observation
    new_obs = np.zeros(obs.shape, dtype=obs.dtype)
    new_obs[s_obs["front_left"]:s_obs["front_left"]+6]   = obs[r_obs["front_left"]:r_obs["front_left"]+6]
    new_obs[s_obs["front_right"]:s_obs["front_right"]+6] = obs[r_obs["front_right"]:r_obs["front_right"]+6]
    new_obs[s_obs["back_left"]:s_obs["back_left"]+6]     = obs[r_obs["back_left"]:r_obs["back_left"]+6]
    new_obs[s_obs["back_right"]:s_obs["back_right"]+6]   = obs[r_obs["back_right"]:r_obs["back_right"]+6]

    # GEN ACTION FROM POLICY
    action, _ = model.predict(np.array([new_obs]), deterministic=True)
    gen_act = action[0]
    assert gen_act.shape == (12,)

    action = np.zeros(gen_act.shape, dtype=gen_act.dtype)
    action[r_act["front_left"]:r_act["front_left"]+3]   = gen_act[s_act["front_left"]:s_act["front_left"]+3]
    action[r_act["front_right"]:r_act["front_right"]+3] = gen_act[s_act["front_right"]:s_act["front_right"]+3]
    action[r_act["back_left"]:r_act["back_left"]+3]     = gen_act[s_act["back_left"]:s_act["back_left"]+3]
    action[r_act["back_right"]:r_act["back_right"]+3]   = gen_act[s_act["back_right"]:s_act["back_right"]+3]

    assert action.shape == (12,)

    # Apply action space clipping
    action_limits = np.array([
        [-np.pi/5, np.pi/5],
        [-0.40,    0.40],
        [0.0,      1.0],
        [-np.pi/5, np.pi/5],
        [-0.40,    0.40],
        [0.0,      1.0],
        [-np.pi/5, np.pi/5],
        [-0.40,    0.40],
        [0.0,      1.0],
        [-np.pi/5, np.pi/5],
        [-0.40,    0.40],
        [0.0,      1.0],
    ], dtype=np.float32)
    action = np.clip(action, action_limits[:,0], action_limits[:,1])

    apply_action(ctrl, action, state)
    t_end = time.time()

    #state["interaction_delays"].append(t_end - t_start)
    
    sleep_time = max(0.01, state["last_time"] + state["dt"] - t_end)
    time.sleep(sleep_time)
    state["last_time"] += state["dt"]


def load_model(agent_file):
    from stable_baselines3 import SAC, PPO
    import stable_baselines3 as sb3
    import pathlib

    model = None
    load_errors = dict()
    for alg in ["SAC", "PPO", "RecurrentPPO"]:
        try:
            if alg == "SAC":
                from stable_baselines3 import SAC
                model = SAC.load(agent_file)
            elif alg == "PPO":
                from stable_baselines3 import PPO
                model = PPO.load(agent_file)
            elif alg == "RecurrentPPO":
                from sb3_contrib import RecurrentPPO
                model = RecurrentPPO.load(agent_file)
        except Exception as e:
            model = None
            load_errors[alg] = e
        if model is not None:
            break

    if model is None:
        lines = [f"Could not trained agent from {agent_file}"]
        for alg, alg_e in load_errors.items():
            lines.append(f"{alg} error: {alg_e}")
        raise RuntimeError("\n".join(lines))

    return model


# Some trajetory functions
def add_to_trajectory(state, kind, content):
    #state["trajectory"].append({"time": time.time(), "kind": kind, "content": content})
    pass

def add_info(state, msg):
    add_to_trajectory(state, kind="info", content=msg)
    print(msg, flush=True)


def run_interactive_loop(walk_file, creep_file, idle_file):
    import torch
    import keyboard
    from .controllers import SpiderController

    model_walk = load_model(walk_file)
    model_creep = load_model(creep_file)
    model_idle = load_model(idle_file)

    state = {
        "gyro": None,
        "velocity": None,
        "last_position": None,
        "dt": 0.25,
        "last_time": None,
        "interaction_delays": [],
        "trajectory": [],
    }

    add_info(state, "Creating controller")
    ctrl = SpiderController()
    ctrl.set_baudrate(4_000_000)
    ctrl.set_duration(1000)
    ctrl.set_acceleration(500)
    ctrl.disable_torque()
    SERVO_ORDER = ctrl.get_servos()

    STANDUP_RADIAN_POS = [
        [0.0, -0.95, 2.35, 0.0, -0.95, 2.35, 0.0, -0.95, 2.35, 0.0, -0.95, 2.35],
        [0.0, -0.00, 1.50, 0.0, -0.00, 1.50, 0.0, -0.00, 1.50, 0.0, -0.00, 1.50],
        [0.0, -0.10, 0.90, 0.0, -0.10, 0.90, 0.0, -0.10, 0.90, 0.0, -0.10, 0.90],
        [0.0, -0.40, 1.05, 0.0, -0.40, 1.05, 0.0, -0.40, 1.05, 0.0, -0.40, 1.05],
        [0.0, -0.65, 1.25, 0.0, -0.65, 1.25, 0.0, -0.65, 1.25, 0.0, -0.65, 1.25],
        [0.0, -0.65, 1.25, 0.0,  0.40, 0.00, 0.0,  0.20, 0.00, 0.0, -0.65, 1.25],
        [0.0,  0.40, 0.00, 0.0,  0.20, 0.00, 0.0,  0.20, 0.00, 0.0,  0.40, 0.00],
        [0.0,  0.20, 0.00, 0.0,  0.20, 0.00, 0.0,  0.20, 0.00, 0.0,  0.20, 0.00],
    ]
    STANDUP_ACTIONS = STANDUP_RADIAN_POS


    def do_standup():
        add_info(state, "Going into standup position")
        ctrl.set_duration(2000)
        ctrl.set_acceleration(1000)
        apply_action(ctrl, np.array(STANDUP_ACTIONS[0]), state)
        time.sleep(2.0)

        ctrl.set_duration(1000)
        ctrl.set_acceleration(500)
        for action in STANDUP_ACTIONS:
            apply_action(ctrl, np.array(action), state, steady=False)
            time.sleep(1.0)

        ctrl.set_duration(500)
        ctrl.set_acceleration(250)
        add_info(state, "NOW STANDING")

    def do_lyingdown():
        add_info(state, "Going into lying down position")
        ctrl.set_duration(2000)
        ctrl.set_acceleration(1000)
        apply_action(ctrl, np.array(STANDUP_ACTIONS[-1]), state)
        time.sleep(2.0)

        ctrl.set_duration(1000)
        ctrl.set_acceleration(500)
        for action in reversed(STANDUP_ACTIONS):
            apply_action(ctrl, np.array(action), state)
            time.sleep(0.5*2)

        add_info(state, "Resetting legs")
        apply_action(ctrl, np.array(STANDUP_ACTIONS[0]), state, steady=False)
        time.sleep(2.0)
        add_info(state, "NOW LYING DOWN")

    scanned_keys = {"up", "down", "left", "right", "space", "esc", "shift"}
    current_state = "inactive"
    while True:
        pressed_keys = {
            k for k in scanned_keys
            if keyboard.is_pressed(k)
        }

        if current_state == "lying_down":
            if "space" in pressed_keys:
                add_info(state, "[TRANSITIONING]: lying_down -> active")
                do_standup()
                current_state = "active"
                add_info(state, "[CURRENT STATE]: active")
            elif "esc" in pressed_keys:
                add_info(state, "[TRANSITIONING]: lying_down -> inactive")
                ctrl.disable_torque()
                current_state = "inactive"
                time.sleep(0.5)
                add_info(state, "[CURRENT STATE]: inactive")
        elif current_state == "active":
            if len(pressed_keys & {"space", "esc"}) == 2:
                add_info(state, "[TRANSITIONING]: active -> inactive")
                ctrl.disable_torque()
                current_state = "inactive"
                time.sleep(0.5)
                add_info(state, "[CURRENT STATE]: inactive")
            elif "space" in pressed_keys:
                add_info(state, "[TRANSITIONING]: active -> lying_down")
                do_lyingdown()
                current_state = "lying_down"
                add_info(state, "[CURRENT STATE]: lying_down")
            elif len(pressed_keys & {"up", "down", "left", "right"}) == 1:
                rot = "0"
                if "up" in pressed_keys: rot = "180"
                elif "down" in pressed_keys: rot = "0"
                elif "left" in pressed_keys: rot = "270"
                elif "right" in pressed_keys: rot = "90"
                fwdmodel = model_walk
                if "shift" in pressed_keys:
                    fwdmodel = model_creep
                step(ctrl, state, fwdmodel, rot=rot)
            else:
                step(ctrl, state, model_idle)
        else:
            if "space" in pressed_keys:
                add_info(state, "[TRANSITIONING]: inactive -> lying_down")
                ctrl.setup_all_servos()
                time.sleep(0.25)
                ctrl.enable_torque()
                time.sleep(0.25)
                add_info(state, "Resetting legs")
                ctrl.set_duration(2000)
                ctrl.set_acceleration(1000)
                apply_action(ctrl, np.array(STANDUP_ACTIONS[0]), state)
                time.sleep(2.0)
                current_state = "lying_down"
                add_info(state, "[CURRENT STATE]: lying_down")
            else:
                time.sleep(0.05)

    print("END OF LOOP, THIS SHOULD NEVER BE REACHED")
