from utils_ import *
from tqdm import tqdm


class MatMulConfig_(MatMulConfig):
    def __init__(self):
        super().__init__()
        self.A_SPARSITY = 0.1
        self.A_SPARSITY_STD = 0.1
        self.B_SPARSITY = 0.1
        self.ALGORITHM = "dense"
        self.N_TEST = 50
        self.USE_PREDICTION = False

        self.matmul_time = []

    def post_process(self):
        if len(self.matmul_time) == 0:
            self.matmul_time_avg = 0
            self.matmul_time_std = 0
        else:
            self.matmul_time_avg = list(np.mean(self.matmul_time, axis=0))
            self.matmul_time_std = list(np.std(self.matmul_time, axis=0))


config = MatMulConfig_()
config.parse_args()

np.random.seed(0)
b_matrix = make_random_sparse_matrix(config.B_HEIGHT, config.B_WIDTH, config.B_SPARSITY)
for i in tqdm(range(config.N_TEST)):
    a_sparsity = np.random.normal(config.A_SPARSITY, config.A_SPARSITY_STD * 2.5)
    if a_sparsity < 0:
        a_sparsity = 0
    elif a_sparsity > 1:
        a_sparsity = 1
    if config.USE_PREDICTION:
        a_matrix = None
    else:
        a_matrix = make_random_sparse_matrix(
            config.A_HEIGHT, config.A_WIDTH, a_sparsity
        )
    sparsity_info = {"input_sparsity": a_sparsity, "weight_sparsity": config.B_SPARSITY}
    matmul_time = matrix_multiplication(
        a_matrix,
        b_matrix,
        config.ALGORITHM,
        config,
        sparsity_info,
        config.USE_PREDICTION,
    )
    config.matmul_time.append(matmul_time)

config.save_config()
