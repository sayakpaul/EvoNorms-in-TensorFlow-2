# Reference
# https://github.com/lonePatient/EvoNorms_PyTorch/blob/master/models/normalization.py

import tensorflow as tf

def group_std(x, groups=32, eps=1e-5):
	N, H, W, C = tf.shape(x)
	x = tf.reshape(x, [N, H, W, groups, C // groups])
	_, var = tf.nn.moments(x, [1, 2, 4], keepdims=True)
	std = tf.sqrt(var + eps)
	std = tf.broadcast_to(std, x.shape)
	return tf.reshape(std, (N, H, W, C))

class EvoNorm2dS0(tf.keras.layers.Layer):
	def __init__(self, in_channels, groups=32, nonlinear=True):
		super(EvoNorm2dS0, self).__init__()
		self.nonlinear = nonlinear
		self.groups = groups

		def build(self):
			self.gamma = self.add_variable("gamma",
									shape=(1, 1, 1, self.in_channels),
									initializer=tf.initializers.Ones())
			self.beta = self.add_variable("beta",
									shape=(1, 1, 1, self.in_channels),
									initializer=tf.initializers.Zeros())
			if self.nonlinear:
				self.v = self.add_variable("v",
									shape=(1, 1, 1, self.in_channels),
									initializer=tf.initializers.Ones())


		def call(self, x):
			if self.nonlinear:
				num = x * tf.nn.sigmoid(self.v * x)
				return num / group_std(x) * self.gamma + self.beta
			else:
				return x * self.gamma + self.beta
