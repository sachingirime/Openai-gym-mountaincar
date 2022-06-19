import gym
import time
env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
from collections import deque
from gym import spaces
import numpy as np

# print("Observation Space: ", env.observation_space)
# print("Action Space       ", env.action_space)


# obs = env.reset()

# for i in range(1000):
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
#     env.render()
#     time.sleep(0.01)
 


# env.close()

# class ConcatObs(gym.Wrapper):
#     def __init__(self, env, k):
#         gym.Wrapper.__init__(self, env)
#         self.k = k
#         self.frames = deque([], maxlen=k)
#         shp = env.observation_space.shape
#         self.observation_space = \
#             spaces.Box(low=0, high=255, shape=((k,) + shp), dtype=env.observation_space.dtype)


#     def reset(self):
#         ob = self.env.reset()
#         for _ in range(self.k):
#             self.frames.append(ob)
#         return self._get_ob()

#     def step(self, action):
#         ob, reward, done, info = self.env.step(action)
#         self.frames.append(ob)
#         return self._get_ob(), reward, done, info

#     def _get_ob(self):
#         return np.array(self.frames)

# env = gym.make("BreakoutNoFrameskip-v4")
# wrapped_env = ConcatObs(env, 4)
# print("The new observation space is", wrapped_env.observation_space)

# # Reset the Env
# obs = wrapped_env.reset()
# print("Intial obs is of the shape", obs.shape)

# # Take one step
# obs, _, _, _  = wrapped_env.step(2)
# print("Obs after taking a step is", obs.shape)


import random 

class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def observation(self, obs):
        # Normalise observation by 255
        return obs / 255.0

class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reward(self, reward):
        # Clip reward between 0 to 1
        return np.clip(reward, 0, 1)
    
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def action(self, action):
        if action == 3:
            return random.choice([0,1,2])
        else:
            return action

# env = gym.make("BreakoutNoFrameskip-v4")
wrapped_env = ObservationWrapper(RewardWrapper(ActionWrapper(env)))

obs = wrapped_env.reset()

for step in range(500):
    action = wrapped_env.action_space.sample()
    obs, reward, done, info = wrapped_env.step(action)
    
    # Raise a flag if values have not been vectorised properly
    if (obs > 1.0).any() or (obs < 0.0).any():
        print("Max and min value of observations out of range")
    
    # Raise a flag if reward has not been clipped.
    if reward < 0.0 or reward > 1.0:
        assert False, "Reward out of bounds"
    
    # Check the rendering if the slider moves to the left.
    wrapped_env.render()
    
    time.sleep(0.001)

wrapped_env.close()

print("All checks passed")

print("Wrapped Env:", wrapped_env)
print("Unwrapped Env", wrapped_env.unwrapped)
print("Getting the meaning of actions", wrapped_env.unwrapped.get_action_meanings())
