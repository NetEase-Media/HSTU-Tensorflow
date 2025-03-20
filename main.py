import os
import time
import argparse
import math
import random
from collections import OrderedDict
import tensorflow as tf
import util

from sampler import WarpSampler
from model import Model
from tqdm import tqdm
from tabulate import tabulate
from util import print_seq_len_percentile, Timer
from evaluator import exact_evaluate
from loguru import logger
import json
import psutil
# from tensorflow.python.profiler import model_analyzer
# from tensorflow.python.profiler import option_builder


def str2bool(s):
    if s not in {'False', 'True', '0', '1'}:
        raise ValueError('Not a valid boolean string')
    if s == 'True' or s == '1':
        return True
    else:
        return False


def str2ints(s):
    values = s.split(',')
    values = tuple(map(int, values))
    return values


def split_by_comma(s):
    return s.split(',')


def add_summary(summary_writer, tag, value, global_step):
    summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)]),
                               global_step=global_step)


def train_epochs(args, dataset, model, sess):
    [user_train, user_valid, _, usernum, itemnum] = dataset
    sampler = WarpSampler(user_train, user_valid, None, usernum, itemnum, batch_size=args.batch_size,
                          maxlen=args.maxlen, n_workers=args.n_workers, args=args)
    num_batch = sampler.length / args.batch_size
    num_batch = math.ceil(num_batch)
    T = 0.0
    t0 = time.time()
    training_reports = []
    summary_writer = tf.summary.FileWriter(args.train_dir, sess.graph)

    with Timer('Training and evaluation'):
        try:
            for epoch in range(1, args.num_epochs + 1):

                for step in tqdm(list(range(num_batch)), total=num_batch, ncols=70, leave=True, unit='b'):
                    u, seq, interval, pos, neg, extra_negs = sampler.next_batch()
                    feed_dict = {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                 model.is_training: True, }
                    if args.load_timestamp:
                        feed_dict[model.input_interval] = interval
                    # run_metadata = tf.RunMetadata()
                    loss, _, summary, global_step = sess.run(
                        [model.loss, model.train_op, model.merged, model.global_step], feed_dict=feed_dict,
                        # options=tf.RunOptions(report_tensor_allocations_upon_oom=True)
                        # options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                        # run_metadata=run_metadata
                    )
                    # profiler.add_step(step=step, run_meta=run_metadata)
                    if step % 50 == 0:
                        summary_writer.add_summary(summary, global_step)

                if epoch % args.eval_every == 0:
                    t1 = time.time() - t0
                    T += t1
                    epoch_report = OrderedDict()
                    exact_t_valid = exact_evaluate(model, dataset, args, sess, mode='valid',
                                                   batch_size=int(args.batch_size * args.pred_batch_size_factor),
                                                   sample_user_num=args.eval_sample_user_num,
                                                   eval_item_not_in_history=args.eval_item_not_in_history)
                    print('')

                    print(
                        'epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f)' % (
                            epoch, T, exact_t_valid[0], exact_t_valid[1]))
                    global_step = sess.run(model.global_step)
                    add_summary(summary_writer, 'valid/NDCG10', exact_t_valid[0], global_step)
                    add_summary(summary_writer, 'valid/HR@10', exact_t_valid[1], global_step)
                    epoch_report['epoch'] = epoch
                    epoch_report['time'] = T
                    epoch_report['valid/NDCG@10'] = exact_t_valid[0]
                    epoch_report['valid/HR@10'] = exact_t_valid[1]
                    if len(exact_t_valid) >= 4:
                        print(
                            'epoch:%d, time: %f(s), valid (NDCG@100: %.4f, HR@100: %.4f)' % (
                                epoch, T, exact_t_valid[2], exact_t_valid[3],))
                        add_summary(summary_writer, 'valid/NDCG100', exact_t_valid[2], global_step)
                        add_summary(summary_writer, 'valid/HR@100', exact_t_valid[3], global_step)
                        epoch_report['valid/NDCG100'] = exact_t_valid[2]
                        epoch_report['valid/HR@100'] = exact_t_valid[3]
                    if len(exact_t_valid) == 6:
                        print(
                            'epoch:%d, time: %f(s), valid (NDCG@200: %.4f, HR@200: %.4f)' % (
                                epoch, T, exact_t_valid[4], exact_t_valid[5],))
                        add_summary(summary_writer, 'valid/NDCG200', exact_t_valid[4], global_step)
                        add_summary(summary_writer, 'valid/HR@200', exact_t_valid[5], global_step)
                        epoch_report['valid/NDCG200'] = exact_t_valid[4]
                        epoch_report['valid/HR@200'] = exact_t_valid[5]

                    training_reports.append(epoch_report)
                    t0 = time.time()
        except Exception:
            import traceback
            traceback.print_exc()
            import sys
            # et, value, tb = sys.exc_info()
            # import pdb
            # pdb.post_mortem(tb)
            sampler.close()
            exit(1)
    sampler.close()
    print("Done training")
    print(tabulate(training_reports, headers='keys', floatfmt='.4f'))
    with open(os.path.join(args.train_dir, 'validation.json'), 'w') as f:
        json.dump(training_reports, f, ensure_ascii=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--remap_hstu_ml_1m', default=False, type=str2bool)
    parser.add_argument('--load_timestamp', default=False, type=str2bool)
    parser.add_argument('--personalize_timestamp_min_diff', default=1, type=int, help='Unit: second')
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--save', default=False, type=str2bool)
    parser.add_argument('--num_epochs', default=201, type=int)
    parser.add_argument('--eval_every', default=20, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--exact_evaluate', default=True, type=str2bool)
    parser.add_argument('--pred_batch_size_factor', default=2, type=float)
    parser.add_argument('--eval_sample_user_num', default=0, type=int)
    parser.add_argument('--eval_item_not_in_history', default=False, type=str2bool)
    parser.add_argument('--n_workers', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--warmup_steps', default=0, type=int)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--embedding_initializer', default='', type=str, choices=('', 'truncated_normal'))
    parser.add_argument('--embedding_scale', default=True, type=str2bool)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--positional_embedding', default=True, type=str2bool)
    parser.add_argument('--inner_size', default=50, type=int)
    parser.add_argument('--inner_act', default='relu', type=str)
    parser.add_argument('--inner_dropout', default=True, type=str2bool)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--context_dropout', default=False, type=str2bool)
    parser.add_argument('--pre_norm', default=True, type=str2bool)
    parser.add_argument('--post_norm', default=False, type=str2bool)
    parser.add_argument('--ffn', default=True, type=str2bool)
    parser.add_argument('--normalize_query', default=True, type=str2bool)
    parser.add_argument('--overwrite_key_with_query', default=True, type=str2bool)
    parser.add_argument('--qkv_projection_initializer', default='None', type=str, choices=('normal', 'None'))
    parser.add_argument('--qkv_projection_bias', default=True, type=str2bool)
    parser.add_argument('--qkv_projection_activation', default='None', type=str)
    parser.add_argument('--value_projection', default=True, type=str2bool)
    parser.add_argument('--attention_type', default='dot_product', type=lambda x: x.split(','))
    parser.add_argument('--compute_hstu_time_interval', default=False, type=str2bool)
    parser.add_argument('--hstu_time_interval_divisor', default=1.0, type=float)
    parser.add_argument('--time_interval_attention_max_interval', default=256, type=int)
    parser.add_argument('--relative_position_bias_add_item_interaction', default=False, type=str2bool)
    parser.add_argument('--scale_attention', default=True, type=str2bool)
    parser.add_argument('--attention_activation', default='None', type=str)
    parser.add_argument('--attention_normalization', default='softmax', type=str)
    parser.add_argument('--time_interval_bias_add_item_interaction', default=True, type=str2bool)
    parser.add_argument('--attention_temperature', default=1.0, type=float)
    parser.add_argument('--attention_kernel', default='relu', type=str, choices=('relu', 'elu'))
    parser.add_argument('--annealing_factor', default=1.0, type=float)
    parser.add_argument('--attention_dropout', default=True, type=str2bool)
    parser.add_argument('--u_projection', default=False, type=str2bool)
    parser.add_argument('--u_projection_initializer', default='None', type=str, choices=('normal', 'None'))
    parser.add_argument('--u_projection_bias', default=True, type=str2bool)
    parser.add_argument('--linear_projection_and_dropout', default=False, type=str2bool)
    parser.add_argument('--dropout_before_linear_projection', default=False, type=str2bool)
    parser.add_argument('--item_bias', default=False, type=str2bool)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--normalize_prediction_embedding', default=False, type=str2bool)
    parser.add_argument('--normalize_test_embedding', default=False, type=str2bool)
    parser.add_argument('--scale_logits', default=0.0, type=float)
    parser.add_argument('--scale_logits_trainable', default=False, type=str2bool)
    parser.add_argument('--scale_logits_trainable_max', default=math.log(100), type=float)
    parser.add_argument('--loss_type', default='bce', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--adam_beta2', default=0.98, type=float)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger.info('')
    logger.info('Start')

    train_dir = args.train_dir
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)
    with open(os.path.join(train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(list(vars(args).items()), key=lambda x: x[0])]))
    f.close()

    with Timer('Load data and preprocess data'):
        dataset = util.load_hstu_ml_1m(args.dataset, args)
        [user_train, user_valid, _, usernum, itemnum] = dataset

    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('Total number of interactions: {}'.format(cc))
    print('Average sequence length: %.2f' % (cc / len(user_train)))
    print_seq_len_percentile(user_train, message='Sequence length percentile: ')

    model = Model(usernum, itemnum, args)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.allow_soft_placement = True
    # config.log_device_placement = True
    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())

    if args.num_epochs > 0:
        train_epochs(args, dataset, model, sess)


if __name__ == '__main__':
    main()
