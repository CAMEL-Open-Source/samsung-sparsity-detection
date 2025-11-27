from utils_ import *


class PeriodConfig(MatMulConfig):
    def __init__(self):
        super().__init__()
        # cpu or cuda:0
        self.SPARSITY_LIST = "/root/workspace/samsungstf/results/matmul_history_bert_small_attention_0_3.json"
        self.OUTPUT_FILE_PATH = "config.json"
        self.N_TEST = 100
        self.DETECT_PERIOD = 1
        self.DETECT_METHOD = "sparsity_based"  # fixed accurate sparsity_based ml_based sparsity_avg_based random
        self.USE_PREDICTION = True
        self.DETECT_PRESET = ""

        self.total_detection_time = 0
        self.total_matmul_time = 0
        self.total_format_conversion_time = 0
        self.total_transpose_time = 0
        self.method_history = []

    def parse_args(self):
        parser = argparse.ArgumentParser(description="GNN Configuration")
        for key, value in self.__dict__.items():
            if key.isupper():
                if isinstance(value, bool):
                    parser.add_argument(f"--{key}", type=str, help=key)
                else:
                    parser.add_argument(f"--{key}", type=type(value), help=key)

        args = parser.parse_args()
        for key, value in vars(args).items():
            if key == "DETECT_PRESET":
                if value == "Fixed":
                    self.DETECT_METHOD = "fixed"
                elif value == "Accurate":
                    self.DETECT_METHOD = "sparsity_based"
                    self.DETECT_PERIOD = 1
                elif value == "Period":
                    self.DETECT_METHOD = "sparsity_based"
                elif value == "Hyper-Predict":
                    self.DETECT_METHOD = "sparsity_avg_based"
                    self.DETECT_PERIOD = int(self.DETECT_PERIOD / 2)
            if value is not None:
                if value == "True":
                    value = True
                elif value == "False":
                    value = False
                setattr(self, key, value)

    def post_process(self):
        self.total_matmul_time = np.sum(
            [history["matmul_time"][1] for history in self.method_history]
        ).item()
        self.total_detection_time = np.sum(
            [history["detection_time"] for history in self.method_history]
        ).item()
        self.total_format_conversion_time = np.sum(
            [history["format_conversion_time"] for history in self.method_history]
        ).item()
        self.total_transpose_time = np.sum(
            [history["transpose_time"] for history in self.method_history]
        ).item()


config = PeriodConfig()
config.parse_args()

sparsity_list = json.load(open(config.SPARSITY_LIST, "r"))

n_skip = len(sparsity_list) // config.N_TEST

last_best_algorithm = None
last_best_algorithm_index = None

idx = 0
current_count = 0
pbar = tqdm(total=config.N_TEST)
while idx < len(sparsity_list):
    sparsity_a = sparsity_list[idx]["input_sparsity"]
    sparsity_b = sparsity_list[idx]["weight_sparsity"]
    a_matrix = make_random_sparse_matrix(config.A_HEIGHT, config.A_WIDTH, sparsity_a)
    b_matrix = make_random_sparse_matrix(config.B_HEIGHT, config.B_WIDTH, sparsity_b)

    if idx == 0:
        # warm up
        get_format_conversion_time(config, a_matrix)

    detection_time = 0
    if last_best_algorithm_index is None or (
        current_count - last_best_algorithm_index >= config.DETECT_PERIOD
        and config.DETECT_PERIOD != -1
    ):
        best_algorithm, detection_time = get_best_algorithm(
            a_matrix, b_matrix, sparsity_list[idx], config
        )
        if config.DETECT_PERIOD == -1:
            best_algorithm = "GEMM"
        last_best_algorithm = best_algorithm
        last_best_algorithm_index = current_count

    mm_time = matrix_multiplication(
        a_matrix,
        b_matrix,
        last_best_algorithm,
        config,
        sparsity_list[idx],
        config.USE_PREDICTION,
    )
    transpose_time = 0
    format_conversion_time = 0
    if "SpMM" in last_best_algorithm:
        if "reverse" in last_best_algorithm:
            format_conversion_time += get_format_conversion_time(config, b_matrix)
        else:
            format_conversion_time += get_format_conversion_time(config, a_matrix)
    if "SpGEMM" in last_best_algorithm:
        format_conversion_time += get_format_conversion_time(config, a_matrix)
        format_conversion_time += get_format_conversion_time(config, b_matrix)
    if "reverse" in last_best_algorithm:
        transpose_time += get_transpose_time(config)
    history = {
        "sparsity_a": sparsity_a,
        "sparsity_b": sparsity_b,
        "algorithm": last_best_algorithm,
        "detection_time": detection_time,
        "matmul_time": mm_time,
        "format_conversion_time": format_conversion_time,
        "transpose_time": transpose_time,
    }
    config.method_history.append(history)
    if config.DEBUG_PRINT:
        print(history)

    idx += n_skip
    current_count += 1
    pbar.update(1)

config.save_config()
