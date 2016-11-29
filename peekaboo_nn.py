import peekaboo_env as peekaboo

import numpy as np
from random import randint
import tensorflow as tf
import matplotlib.pyplot as plt

class TfLearner:
	def __init__(self, env):
		self.env = env

	def _one_hot(self, size, col):
		return np.identity(size)[col:col+1]

	def _one_hot_state(self, state):
		return self._one_hot(self.env.nS, state)

	def learn(self, gamma=0.99, epsilon=0.1, episodes=2000, max_itr=99):
		tf.reset_default_graph()

		# Build feed-forward part of the network used to choose actions.
		inputs1 = tf.placeholder(shape=[1, 	self.env.nS], dtype=tf.float32)
		W = tf.Variable(tf.random_uniform([self.env.nS, self.env.nA], 0, 0.01))
		Qout = tf.matmul(inputs1, W)
		predict = tf.argmax(Qout, 1)

		# Compute loss which is sum of squares difference between target and
		# predicted Q values.
		nextQ = tf.placeholder(shape=[1, self.env.nA], dtype=tf.float32)
		loss = tf.reduce_sum(tf.square(nextQ - Qout))
		trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
		updateModel = trainer.minimize(loss)

		# Train the network.
		init = tf.initialize_all_variables()

		# Create lists.
		jList = []
		rList = []
		with tf.Session() as sess:
			sess.run(init)
			for i in range(episodes):
				print 'Running episode:{}'.format(i)
				self.env.reset()
				j = 0
				rAll = 0
				# The Q-network.
				while j < max_itr:
					j+= 1
					# Choose an action greedily with epsilon chance of random action from the 
					# Q network.
					action, allQ = sess.run([predict, Qout],
						feed_dict={inputs1:self._one_hot_state(self.env.s)})
					action = np.asscalar(action)
					if np.random.rand(1) < epsilon:
						action = self.env.pick_random_action()

					# Get the new state and reward from the environment.
					s_prime, reward, done, _ = self.env.step(action)
					# Obtain the Qprime values by feeding the new state through the network.
					Qprime = sess.run(Qout,
						feed_dict={inputs1:self._one_hot_state(s_prime)})
					maxQPrime = np.max(Qprime)
					targetQ = allQ
					targetQ[0, action] = reward + gamma * maxQPrime	
					# Train the network using target and predicted Qvalues.
					_, W1 = sess.run([updateModel, W],
						feed_dict={inputs1:self._one_hot_state(self.env.s),nextQ:targetQ})
					rAll += reward
					if done:
						epsilon = 1./((1/50) + 10)
						break
				jList.append(j)
				rList.append(rAll)
				print "Percent of successful episodes: {}%".format(sum(rList) / episodes)
		return rList, jList

def tf_qtable_policy():
	ring_size = 10
	peekaboo_env = peekaboo.PeekabooHiddenEnv(ring_size=ring_size)
	learner = TfLearner(env=peekaboo_env)
	rewards, jList = learner.learn()
	plt.plot(rewards)
	plt.show()
	plt.plot(jList)
	plt.show()

# fixed_action_policy()

tf_qtable_policy()
