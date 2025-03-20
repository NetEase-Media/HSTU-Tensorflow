# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

from __future__ import print_function

import json
import math
import os
import random

import tensorflow as tf
import numpy as np
from tqdm import tqdm


def positional_encoding(dim, sentence_length, dtype=tf.float32):

    encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])

    return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)


def normalize(inputs,
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.
    
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs


def embedding_lookup(table, inputs: tf.Tensor):
    shape = inputs.get_shape().as_list()[1:] + table.get_shape().as_list()[-1:]
    sparse = tf.contrib.layers.dense_to_sparse(inputs, -1)
    output = tf.nn.embedding_lookup_sparse(table, sparse, None)
    output = tf.reshape(output, [-1, ] + shape)
    return output


def embedding(inputs, 
              vocab_size, 
              num_units, 
              zero_pad=True,
              initializer='',
              initializer_stddev=0.02,
              scale=True,
              l2_reg=0.0,
              scope="embedding", 
              with_t=False,
              reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.
        
    For example,
    
    ```
    import tensorflow as tf
    
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]

     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```
    
    ```
    import tensorflow as tf
    
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]

     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]    
    ```    
    '''
    with tf.variable_scope(scope, reuse=reuse):
        if initializer == 'truncated_normal':
            lookup_table = tf.get_variable('lookup_table',
                                           dtype=tf.float32,
                                           shape=[vocab_size, num_units],
                                           initializer=tf.initializers.truncated_normal(stddev=initializer_stddev))
        else:
            lookup_table = tf.get_variable('lookup_table',
                                           dtype=tf.float32,
                                           shape=[vocab_size, num_units],
                                           #initializer=tf.contrib.layers.xavier_initializer(),
                                           regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        else:
            lookup_table = tf.concat((lookup_table, tf.zeros(shape=[1, num_units]),), 0)

        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        # outputs = embedding_lookup(lookup_table, inputs)
        
        if scale:
            outputs = outputs * (num_units ** 0.5) 
    if with_t: return outputs,lookup_table
    else: return outputs


def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


def clip_by_value_preserve_gradient(t, clip_value_min, clip_value_max,
                                    name=None):
    """Clips values to a specified min and max while leaving gradient unaltered.

    Like `tf.clip_by_value`, this function returns a tensor of the same type and
    shape as input `t` but with values clamped to be no smaller than to
    `clip_value_min` and no larger than `clip_value_max`. Unlike
    `tf.clip_by_value`, the gradient is unaffected by this op, i.e.,

    ```python
    tf.gradients(tfp.math.clip_by_value_preserve_gradient(x), x)[0]
    # ==> ones_like(x)
    ```

    Note: `clip_value_min` needs to be smaller or equal to `clip_value_max` for
    correct results.

    Args:
      t: A `Tensor`.
      clip_value_min: A scalar `Tensor`, or a `Tensor` with the same shape
        as `t`. The minimum value to clip by.
      clip_value_max: A scalar `Tensor`, or a `Tensor` with the same shape
        as `t`. The maximum value to clip by.
      name: A name for the operation (optional).
        Default value: `'clip_by_value_preserve_gradient'`.

    Returns:
      clipped_t: A clipped `Tensor`.
    """
    with tf.name_scope(name or 'clip_by_value_preserve_gradient'):
        t = tf.convert_to_tensor(t, name='t')
        clip_t = tf.clip_by_value(t, clip_value_min, clip_value_max)
        return t + tf.stop_gradient(clip_t - t)


def relative_position_bias(batch_size, max_len, num_heads, scope, reuse):
    with tf.variable_scope(scope, reuse=reuse):
        relative_position = np.expand_dims(np.arange(max_len), axis=1) - np.expand_dims(np.arange(max_len), axis=0)
        relative_position += max_len
        relative_position_embeddings = embedding(relative_position, 2 * max_len, 1, zero_pad=False,
                                                 initializer='truncated_normal', initializer_stddev=0.01, scale=False,
                                                 scope='relative_position_embeddings')
        relative_position_embeddings = tf.expand_dims(relative_position_embeddings, axis=0)
        bias = tf.squeeze(relative_position_embeddings, axis=-1)
        bias = tf.tile(bias, [batch_size * num_heads, 1, 1])
        return bias


def time_interval_bias(input_interval, max_len, max_interval, num_heads, scope, reuse):
    with tf.variable_scope(scope, reuse=reuse):
        relative_position = np.expand_dims(np.arange(max_len), axis=1) - np.expand_dims(np.arange(max_len), axis=0)
        relative_position += max_len
        relative_position_embeddings = embedding(relative_position, 2 * max_len, 1, zero_pad=False,
                                                 initializer='truncated_normal', initializer_stddev=0.01, scale=False,
                                                 scope='relative_position_embeddings')
        relative_position_embeddings = tf.expand_dims(relative_position_embeddings, axis=0)
        interval_embeddings = embedding(input_interval, max_interval + 1, 1, zero_pad=False, initializer='truncated_normal',
                                        initializer_stddev=0.01, scale=False, scope='time_interval_embeddings')
        bias = relative_position_embeddings + interval_embeddings
        bias = tf.squeeze(bias, axis=-1)
        bias = tf.tile(bias, [num_heads, 1, 1])
        return bias


def crop_to_batch(a, offset_height, offset_width, target_height, target_width, dtype=tf.float32):
    def crop_fn(args):
        i, j = args
        return tf.slice(a, [i, j], [target_height, target_width])
    return tf.map_fn(crop_fn, (offset_height, offset_width), dtype=dtype)


def silu(x):
    return x * tf.nn.sigmoid(x)


def apply_attention(K_, V_, causality, dropout_rate, is_training, keys, num_heads, attention, queries,
                    scale_attention=True, padding_value=-2 ** 32 + 1, attention_activation=None,
                    attention_normalization='softmax', attention_temperature=1.0, args=None, scope=None):
    with tf.name_scope(scope) as scope:
        # Scale
        if scale_attention:
            print('Scale attention')
            attention = attention / (K_.get_shape().as_list()[-1] ** 0.5)
        # Key Masking
        key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)
        paddings = tf.ones_like(attention) * padding_value
        attention = tf.where(tf.equal(key_masks, 0), paddings, attention)  # (h*N, T_q, T_k)
        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(attention[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(attention)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * padding_value
            attention = tf.where(tf.equal(masks, 0), paddings, attention)  # (h*N, T_q, T_k)
        # Activation
        if attention_activation == 'silu':
            print('Use SiLU attention activation')
            attention = silu(attention)

        if attention_normalization == 'softmax':
            if attention_temperature != 1.0:
                attention = attention / attention_temperature
            attention = tf.nn.softmax(attention)  # (h*N, T_q, T_k)
        elif attention_normalization == 'sum':
            attention = attention / tf.reduce_sum(attention + 1e-6, axis=-1, keep_dims=True)
        elif attention_normalization == 'max_length':
            print('Normalize attention score by max length')
            attention = attention / tf.cast(tf.shape(attention)[-1], tf.float32)
        elif attention_normalization == 'real_length':
            print('Normalize attention score by real length')
            real_length = tf.cast(tf.reduce_sum(
                key_masks * masks if causality else key_masks, axis=-1, keep_dims=True), tf.float32, name='real_length')
            real_length += 1e-9
            attention = attention / real_length

        elif attention_normalization == 'None':
            print('Do not normalize attention score')
        else:
            raise ValueError

        # Query Masking
        query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        attention *= query_masks  # broadcasting. (N, T_q, C)

        tf.summary.image('{}/attention_activation'.format(scope), tf.expand_dims(attention, axis=-1))
        tf.summary.image('{}/attention_activation_row_normalized'.format(scope),
                         tf.expand_dims(attention / (tf.reduce_max(attention, keep_dims=True, axis=-1) + 1e-6), axis=-1))
        # Dropouts
        if args.attention_dropout:
            attention = tf.layers.dropout(attention, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Weighted sum
        new_value = tf.matmul(attention, V_)  # ( h*N, T_q, C/h)
        return new_value


def multihead_attention(queries, 
                        keys,
                        input_interval=None,
                        num_units=None, 
                        num_heads=8,
                        attention_type='dot_product',
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        linear_projection_and_dropout=False,
                        args=None,
                        scope="multihead_attention", 
                        reuse=None,
                        with_qk=False):
    '''Applies multihead attention.
    
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        if args.normalize_query:
            queries = normalize(queries)
        if args.overwrite_key_with_query:
            keys = queries

        if args.qkv_projection_initializer == 'normal':
            print('Set qkv projection initializer to normal')
            qkv_projection_initializer = lambda: tf.random_normal_initializer(0, 0.02)
        else:
            qkv_projection_initializer = None

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=None, use_bias=args.qkv_projection_bias,
                            kernel_initializer=qkv_projection_initializer())  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=None, use_bias=args.qkv_projection_bias,
                            kernel_initializer=qkv_projection_initializer())  # (N, T_k, C)
        if args.value_projection:
            V = tf.layers.dense(keys, num_units, activation=None, use_bias=args.qkv_projection_bias,
                                kernel_initializer=qkv_projection_initializer())  # (N, T_k, C)
        else:
            V = keys
        batch_size = tf.shape(V)[0]

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        if args.qkv_projection_activation == 'silu':
            print('Use SiLU activation on qkv projection')
            Q_, K_, V_ = silu(Q_), silu(K_), silu(V_)

        new_values = 0
        if 'dot_product' in attention_type:
            print('Add dot product attention')
            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
            outputs = apply_attention(K_, V_, causality, dropout_rate, is_training, keys, num_heads, outputs,
                                      queries, args=args, scope='apply_dot_product_attention')
            new_values += outputs

        if 'relative_position_bias' in attention_type:
            print('Add relative position bias')
            attention_bias = relative_position_bias(batch_size, args.maxlen, num_heads, scope='relative_position_bias',
                                                    reuse=reuse)
            if args.relative_position_bias_add_item_interaction:
                print('Relative position bias add item interaction')
                outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
                outputs += attention_bias
            else:
                outputs = attention_bias

            outputs = apply_attention(K_, V_, causality, dropout_rate, is_training, keys, num_heads, outputs,
                                      queries, scale_attention=args.scale_attention,
                                      attention_activation=args.attention_activation,
                                      attention_normalization=args.attention_normalization, args=args,
                                      scope='relative_position_bias_attention')

            new_values += outputs

        if 'time_interval_bias' in attention_type:
            print('Add time interval bias attention')
            attention_bias = time_interval_bias(input_interval, args.maxlen, args.time_interval_attention_max_interval,
                                                num_heads, scope='time_interval_bias', reuse=reuse)
            if args.time_interval_bias_add_item_interaction:
                outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
                outputs += attention_bias
            else:
                outputs = attention_bias
            outputs = apply_attention(K_, V_, causality, dropout_rate, is_training, keys, num_heads, outputs,
                                      queries, scale_attention=args.scale_attention,
                                      attention_activation=args.attention_activation,
                                      attention_normalization=args.attention_normalization, args=args,
                                      scope='apply_time_interval_bias_attention')
            new_values += outputs

        outputs = tf.concat(tf.split(new_values, num_heads, axis=0), axis=2 ) # (N, T_q, C)

        if args.u_projection:
            if args.u_projection_initializer == 'normal':
                u_projection_initializer = lambda: tf.random_normal_initializer(0, 0.02)
            else:
                u_projection_initializer = None
            U = tf.layers.dense(queries, num_units, activation=None, use_bias=args.u_projection_bias,
                                kernel_initializer=u_projection_initializer())
            U = silu(U)
            outputs = U * normalize(outputs)

        if linear_projection_and_dropout:
            if args.dropout_before_linear_projection:
                outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
            outputs = tf.layers.dense(outputs, num_units, activation=None, kernel_initializer=qkv_projection_initializer())
            if not args.dropout_before_linear_projection:
                outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Residual connection
        outputs += queries
              
        # Normalize
        #outputs = normalize(outputs) # (N, T_q, C)
 
    if with_qk: return Q,K
    else: return outputs


def get_activation_function(activation_name: str):
    activation_map = {
        'relu': tf.nn.relu,
        'gelu': lambda x: 0.5 * x * (1.0 + tf.math.erf(x / 1.4142135623730951))
    }
    return activation_map[activation_name.lower()]


def feedforward(inputs, 
                num_units=[2048, 512],
                inner_act='relu',
                scope="multihead_attention", 
                dropout_rate=0.2,
                inner_dropout=True,
                is_training=True,
                reuse=None):
    '''Point-wise feed forward net.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": get_activation_function(inner_act), "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        if inner_dropout:
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        
        # Residual connection
        outputs += inputs
        
        # Normalize
        #outputs = normalize(outputs)
    
    return outputs
