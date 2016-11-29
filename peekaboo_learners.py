# https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.eqlpnna9k

import peekaboo_env as peekaboo

import numpy as np
from random import randint

def fixed_action(peekaboo_env, action=peekaboo.LEFT):
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
	print 'Only peekaboo.LEFT'
	peekaboo_env = peekaboo.PeekabooHiddenEnv(ring_size=10)
	fixed_action(peekaboo_env=peekaboo_env, action=peekaboo.LEFT)

	print ''
	print 'Only peekaboo.RIGHT'
	peekaboo_env.reset()
	fixed_action(peekaboo_env=peekaboo_env, action=peekaboo.RIGHT)

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

	def learn(self, max_itr, episodes=1000, lr=0.8, gamma=0.7, break_on_done=False):
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
	peekaboo_env = peekaboo.PeekabooHiddenEnv(ring_size=ring_size)
	learner = QTableLearner(env=peekaboo_env)
	max_itr = int(ring_size * 0.75)
	learner.learn(max_itr=max_itr, break_on_done=False)
	learner.print_q_table()

fixed_action_policy()

qtable_policy()
