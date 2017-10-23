import argparse
import Go_Env
from model import alphagozero_resent_model

if __name__=="__main__":


    parser = argparse.ArgumentParser(description='Define parameters.')

    parser.add_argument('--n_epoch', type=int, default=10)
    parser.add_argument('--n_batch', type=int, default=64)
    parser.add_argument('--n_img_row', type=int, default=32)
    parser.add_argument('--n_img_col', type=int, default=32)
    parser.add_argument('--n_img_channels', type=int, default=3)
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--n_resid_units', type=int, default=5)
    parser.add_argument('--lr_schedule', type=int, default=60)
    parser.add_argument('--lr_factor', type=float, default=0.1)

    args = parser.parse_args()
    
    run = Go_Env.Supervised_Learning_Env()

    hps = alphagozero_resent_model.HParams(batch_size=run.batch_num,
                               num_classes=run.nb_classes,
                               min_lrn_rate=0.0001,
                               lrn_rate=args.lr,
                               num_residual_units=args.n_resid_units,
                               use_bottleneck=False,
                               weight_decay_rate=0.0002,
                               relu_leakiness=0.1,
                               optimizer='mom')

    run.train(hps)
