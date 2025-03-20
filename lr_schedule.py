import tensorflow as tf


def lr_warmup(global_step,
              warmup_steps,
              start_lr=0.0,
              target_lr=1e-3):
    with tf.name_scope('lr_warmup'):
        # Target LR * progress of warmup (=1 at the final warmup step)
        warmup_lr = start_lr + (target_lr - start_lr) * (tf.to_float(global_step) / tf.to_float(warmup_steps))

        learning_rate = tf.where(global_step < warmup_steps, warmup_lr, target_lr)
        return learning_rate
