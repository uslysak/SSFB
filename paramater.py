import argparse

def parameter_parser():

    parser = argparse.ArgumentParser(description="Run SeGNE.")

    # ----------  随机游走 ----------
    parser.add_argument("--P",
                        type=float,
                        default=1,
                        help="Return hyperparameter. Default is 1.")

    parser.add_argument("--Q",
                        type=float,
                        default=8,
                        help="In-out hyperparameter. Default is 1.")

    parser.add_argument("--random-walk-length",
                        type=int,
                        default=80,
                        help="Length of random walk per source. Default is 80.")

    parser.add_argument("--num-of-walks",
                        type=int,
                        default=5,
                        help="Number of random walks per source. Default is 5.")

    parser.add_argument("--window-size",
                        type=int,
                        default=5,
                        help="Window size for proximity statistic extraction. Default is 5.")

    # ---------- embedding  ----------
    parser.add_argument("--dimensions",
                        type=int,
                        default=64,
                        help="Number of dimensions. Default is 16.")

    parser.add_argument("--distortion",
                        type=float,
                        default=0.75,
                        help="Downsampling distortion. Default is 0.75.")

    parser.add_argument("--negative-sample-number",
                        type=int,
                        default=10,
                        help="Number of negative samples to draw. Default is 10.")

    # ---------- 学习率调度 ----------
    parser.add_argument("--initial-learning-rate",
                        type=float,
                        default=0.01,
                        help="Initial learning rate. Default is 0.01.")

    parser.add_argument("--minimal-learning-rate",
                        type=float,
                        default=0.001,
                        help="Minimal learning rate. Default is 0.001.")

    parser.add_argument("--annealing-factor",
                        type=float,
                        default=1,
                        help="Annealing factor. Default is 1.0.")

    # ---------- gamma 调度 ----------
    parser.add_argument("--initial-gamma",
                        type=float,
                        default=0.1,
                        help="Initial clustering weight. Default is 0.1.")

    parser.add_argument("--final-gamma",
                        type=float,
                        default=0.1,
                        help="Final clustering weight. Default is 0.1.")

    # ---------- 聚类数 ----------
    parser.add_argument("--cluster-number",
                        type=int,
                        default=4,
                        help="Number of clusters. Default is 4.")

    return parser.parse_args()