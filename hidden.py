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

def fixed_action(peekaboo_env, action=LEFT):
	print '{} [ball:{}] Start:{}'.format(peekaboo_env.hidden_env.s,
		peekaboo_env.ball_state,
		peekaboo_env.current_state_str())
	while True:
		if peekaboo_env.done():
			print 'Done'
			break
		peekaboo_env.step(action)
		print '{} Next:{}'.format(peekaboo_env.hidden_env.s, peekaboo_env.current_state_str())


def fixed_action_policy():
	print 'Only Left'
	peekaboo_env = PeekabooHiddenEnv(ring_size=10)
	fixed_action(peekaboo_env=peekaboo_env, action=LEFT)

	print ''
	print 'Only Right'
	peekaboo_env.reset()
	fixed_action(peekaboo_env=peekaboo_env, action=RIGHT)

class QTableLearner:
	def __init__(self, env):		
		self.peekaboo_env = env
		self.Q = np.zeros((self.peekaboo_env.nS, self.peekaboo_env.nA))

	def print_q_table(self):
		action = '   '.join([self.peekaboo_env.action_str(a) for a in xrange(self.peekaboo_env.nA)])
		print '   {}'.format(action)

		for s in xrange(self.peekaboo_env.nS):
			state_str = self.peekaboo_env.state_str(s)
			print '{} {}'.format(state_str, self.Q[s,:])

	def learn(self, max_itr, episodes=1000, lr=0.8, gamma=0.3, break_on_done=False):
		for i in xrange(episodes):
			self.peekaboo_env.reset()
			print '{} [ball:{}] Start:{}'.format(self.peekaboo_env.hidden_env.s,
				self.peekaboo_env.ball_state,
				self.peekaboo_env.current_state_str())
			self._run_episode(i, max_itr, lr, gamma, break_on_done)
		print ''
		print 'Done learning'

	def _pick_action(self, episode_num, rand=True):
		if rand:
			return self.peekaboo_env.pick_random_action()
		s = self.peekaboo_env.s
		nA = self.peekaboo_env.nA
		return np.argmax(self.Q[s,:] + np.random.randn(1, nA) * (1.0 / (episode_num + 1)))

	def _run_episode(self, max_itr, episode_num, lr, gamma, break_on_done):
		print ''
		print 'Running episode:{}'.format(episode_num)
		i = 0
		while True:
			# Given current state, pick an action.
			action = self._pick_action(episode_num)
			s = self.peekaboo_env.s

			# Transition to the new state by taking the action.
			next_state, reward, is_done, _ = self.peekaboo_env.step(action)
			print '{} Next:{}'.format(self.peekaboo_env.hidden_env.s,
				self.peekaboo_env.current_state_str())

			s_q_val = reward + gamma * np.max(self.Q[next_state,:])
			self.Q[s][action] += lr * (s_q_val - self.Q[s][action])
			i += 1
			if i > max_itr:
				print '    MaxItr reached. Done with episode'
				return
			if break_on_done and self.peekaboo_env.done():
				print '    Breaking on done. Done with episode'
				return

def qtable_policy():
	ring_size = 10
	peekaboo_env = PeekabooHiddenEnv(ring_size=ring_size)
	learner = QTableLearner(env=peekaboo_env)
	max_itr = int(ring_size * 0.75)
	learner.learn(max_itr=max_itr, break_on_done=False)
	learner.print_q_table()

# fixed_action_policy()

qtable_policy()
