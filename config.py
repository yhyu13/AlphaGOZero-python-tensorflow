import argparse
from collections import namedtuple


parser = argparse.ArgumentParser(description='Define parameters.')

"""Network hyperparameters"""
parser.add_argument('--n_epoch', type=int, default=1)
parser.add_argument('--global_epoch', type=int, default=50)
parser.add_argument('--n_batch', type=int, default=16)
parser.add_argument('--n_img_row', type=int, default=19)
parser.add_argument('--n_img_col', type=int, default=19)
parser.add_argument('--n_img_channels', type=int, default=17)
parser.add_argument('--n_classes', type=int, default=19**2+1)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--n_resid_units', type=int, default=6)
parser.add_argument('--n_gpu', type=int, default=1)
parser.add_argument('--dataset', dest='processed_dir',default='./processed_data')
parser.add_argument('--model_path',dest='load_model_path',default='./savedmodels')#'./savedmodels'
parser.add_argument('--model_type',dest='model',default='full',\
                    help='choose residual block architecture {original,elu,full}')
parser.add_argument('--optimizer',dest='opt',default='mom')

"""Go Text Protocol Policy Network Type"""
parser.add_argument('--gtp_policy',dest='gpt_policy',default='mctspolicy',help='choose gtp bot player')#random,mctspolicy

"""Self Play Pipeline"""
parser.add_argument('--num_playouts',type=int,dest='num_playouts',default=1600,help='The number of MC search per move, the more the better.')
parser.add_argument('--selfplay_games_per_epoch',type=int,dest='selfplay_games_per_epoch',default=25000)
parser.add_argument('--selfplay_games_against_best_model',type=int,dest='selfplay_games_against_best_model',default=400)

"""Which main function to excute"""
parser.add_argument('--mode',dest='MODE', default='train',help='among selfplay, gtp and train')

FLAGS = parser.parse_args()



"""ResNet hyperparameters"""
HParams = namedtuple('HParams',
                 'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                 'num_residual_units, use_bottleneck, weight_decay_rate, '
                 'relu_leakiness, optimizer, temperature, global_norm, num_gpu, '
                 'name')

HPS = HParams(batch_size=FLAGS.n_batch,
               num_classes=FLAGS.n_classes,
               min_lrn_rate=0.0001,
               lrn_rate=FLAGS.lr,
               num_residual_units=FLAGS.n_resid_units,
               use_bottleneck=False,
               weight_decay_rate=0.0001,
               relu_leakiness=0,
               optimizer=FLAGS.opt,
               temperature=1.0,
               global_norm=100,
               num_gpu=FLAGS.n_gpu,
               name='01')
