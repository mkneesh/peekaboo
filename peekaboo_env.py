import numpy as np
from random import randint

from gym import Env
from gym.envs.toy_text import discrete

LEFT = 0
RIGHT = 1
STAY = 2

MAP_ACTIONS = {}
MAP_ACTIONS[LEFT]   = "LEFT"
MAP_ACTIONS[RIGHT]  = "RIGHT"
MAP_ACTIONS[STAY]   = "STAY"

def make_hidden_env(ring_size=10, done_reward=100):
	# [0,ring_size)
	ball_in = randint(0,ring_size-1)

	done = lambda s: s == ball_in
	reward = lambda s : done_reward if done(s) else 0

	def transition(to_state):
		"""P[s][a] == [(probability, nextstate, reward, done), ...]"""
		return (1.0, to_state, reward(to_state), done(to_state))

	nS = ring_size
	nA = 3
	P = {}
	for s in xrange(ring_size):
		next_state = s
		P[s] = {}
		P[s][STAY] = [transition(next_state)]

		next_state = (s + 1) % ring_size
		P[s][RIGHT] = [transition(next_state)]

		next_state = (s - 1) if (s - 1) >= 0 else ring_size - 1
		P[s][LEFT] = [transition(next_state)]

	isd = np.ones(nS) * (1.0 / nS)
	return discrete.DiscreteEnv(nS, nA, P, isd), ball_in

NOT_IN_SIGHT = 0   # 0 0 0
LEFT_IN_SIGHT = 1  # 1 0 0
IN_SIGHT = 2       # 0 1 0
RIGHT_IN_SIGHT = 3 # 0 0 1

MAP_STATES = {}
MAP_STATES[NOT_IN_SIGHT]   = "0 0 0"
MAP_STATES[LEFT_IN_SIGHT]  = "1 0 0"
MAP_STATES[IN_SIGHT]       = "0 1 0"
MAP_STATES[RIGHT_IN_SIGHT] = "0 0 1"

class PeekabooHiddenEnv(Env):
	def __init__(self, ring_size=10, done_reward=100):
		self.nS = 4
		self.nA = 3
		self.ring_size = ring_size
		self.done_reward = done_reward

		self.ball_state = None
		self.hidden_env = None
		self.s = None
		self._reset()  # Populates the above Nones.

	def _reset(self):
		self.hidden_env, self.ball_state = make_hidden_env(
			self.ring_size, self.done_reward)	
		self.s = self._hidden_to_observed(self.hidden_env.s)

	def _hidden_to_observed(self, hidden):
		"""Returns the observed state given the hidden state."""
		if self.ball_state == hidden:
			return IN_SIGHT
		if ((hidden + 1) % self.hidden_env.nS) == self.ball_state:
			return RIGHT_IN_SIGHT
		if ((self.ball_state + 1) % self.hidden_env.nS) == hidden:
			return LEFT_IN_SIGHT
		return NOT_IN_SIGHT

	def _step(self, a):
		hidden_state, reward, is_done, debug = self.hidden_env.step(a)
		self.s = self._hidden_to_observed(hidden_state)
		return (self.s, reward, is_done, debug)

	def _seed(self, seed=None):
		return self.hidden_env.seed(seed)

	def done(self):
		return self.s == IN_SIGHT

	def current_state_str(self):
		return MAP_STATES[self.s]

	def state_str(self, s):
		return MAP_STATES[s]

	def action_str(self, action):
		return MAP_ACTIONS[action]

	def pick_random_action(self):
		return randint(0, self.nA-1)