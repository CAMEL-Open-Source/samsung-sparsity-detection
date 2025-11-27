from utils_ import *


class MatMulConfig_(MatMulConfig):
    def __init__(self):
        super().__init__()
        self.A_SPARSITY = 0.1
        self.B_SPARSITY = 0.1
        self.ALGORITHM = "GEMMTile"

        self.matmul_time = [0, 0, 0, 0]

    def post_process(self):
        pass


config = MatMulConfig_()
config.parse_args()

a_matrix = make_random_sparse_matrix(config.A_HEIGHT, config.A_WIDTH, config.A_SPARSITY)
b_matrix = make_random_sparse_matrix(config.B_HEIGHT, config.B_WIDTH, config.B_SPARSITY)

config.matmul_time = matrix_multiplication(a_matrix, b_matrix, config.ALGORITHM, config)

config.save_config()
