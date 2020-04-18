# Reference
# https://github.com/lonePatient/EvoNorms_PyTorch/blob/master/models/normalization.py

import tensorflow as tf

def instance_std(x, eps=1e-5):
	# https://www.tensorflow.org/api_docs/python/tf/nn/moments
	_, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
	return tf.sqrt(var + eps)

class EvoNorm2dB0(tf.keras.layers.Layer):
	def __init__(self, in_channels, nonlinear=True, momentum=0.9,
		eps=1e-5):
		super(EvoNorm2dB0, self).__init__()
		self.nonlinear = nonlinear
		self.momentum = momentum
		self.eps = eps
		self.running_var = tf.ones((1, in_channels, 1, 1))

		def build(self):
			self.gamma = self.add_variable("gamma",
									shape=(1, self.in_channels, 1, 1),
									initializer=tf.initializers.Ones())
			self.beta = self.add_variable("beta",
									shape=(1, self.in_channels, 1, 1),
									initializer=tf.initializers.Zeros())
			if self.nonlinear:
				self.v = self.add_variable("v",
									shape=(1, self.in_channels, 1, 1),
									initializer=tf.initializers.Ones())


		def call(self, x):
			N, H, W, C = tf.shape(x)

			if self.training:
				x1 = tf.transpose(x, [3, 0, 1, 2])
				x1 = tf.reshape(x1, (C, -1))
				var = tf.math.reduce_std(x1, axis=1)
				var = tf.reshape(var, (1, C, 1, 1))
				self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
			else:
				var = self.running_var

			if self.nonlinear:
				den = tf.math.maximum(tf.sqrt(var+self.eps), self.v * x + instance_std(x))
				return x / den * self.gamma + self.beta
			else:
				return x * self.gamma + self.beta