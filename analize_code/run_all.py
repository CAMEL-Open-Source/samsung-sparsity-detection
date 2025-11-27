from benchmark.run_shell import *
from benchmark.collector import *
import glob

RESULT_DIR = "/root/workspace/samsungstf/training_results"


def run_resnet18():
    # get /root/workspace/samsungstf/training_results/resnet18_*.pth
    resnet18_files = glob_files(RESULT_DIR, "resnet18_\d+.pth")
    resnet18_files = [file.replace(RESULT_DIR + "/", "") for file in resnet18_files]

    cmd = f"python /root/workspace/samsungstf/analize_code/get_sparsity_resnet.py {{0}}"

    cmd_list = []
    temp_list = format_matrix_cmd(cmd, resnet18_files)
    cmd_list += temp_list

    run_benchmark(cmd_list, 1, False)


def run_bert():
    # get /root/workspace/samsungstf/training_results/resnet18_*.pth
    resnet18_files = glob_files(RESULT_DIR, "bert_\d+.pth")
    resnet18_files = [file.replace(RESULT_DIR + "/", "") for file in resnet18_files]

    cmd = f"python /root/workspace/samsungstf/analize_code/get_sparsity_bert.py {{0}}"

    cmd_list = []
    temp_list = format_matrix_cmd(cmd, resnet18_files)
    cmd_list += temp_list

    run_benchmark(cmd_list, 8, False)


def run_gpt():
    # get /root/workspace/samsungstf/training_results/resnet18_*.pth
    resnet18_files = glob_files(RESULT_DIR, "gpt2_\d+.pth")
    resnet18_files = [file.replace(RESULT_DIR + "/", "") for file in resnet18_files]

    cmd = f"python /root/workspace/samsungstf/analize_code/get_sparsity_gpt.py {{0}}"

    cmd_list = []
    temp_list = format_matrix_cmd(cmd, resnet18_files)
    cmd_list += temp_list

    run_benchmark(cmd_list, 8, False)


def run_diffusion():
    # get /root/workspace/samsungstf/training_results/resnet18_*.pth
    resnet18_files = glob_files(RESULT_DIR, "diffusion_relu_\d+.pth")
    resnet18_files = [file.replace(RESULT_DIR + "/", "") for file in resnet18_files]

    cmd = f"python /root/workspace/samsungstf/analize_code/get_sparsity_diffusion.py {{0}}"

    cmd_list = []
    temp_list = format_matrix_cmd(cmd, resnet18_files)
    cmd_list += temp_list

    run_benchmark(cmd_list, 2, False)


def run_optimal_operator_gain():
    target_size = "1024"
    target_algorithms = "GEMM SpMMRowR SpMMRowC SpMMElemR SpMMElemC SpGEMMHashR SpGEMMHashC SpGEMMSortR SpGEMMSortC"
    target_sparsity = "0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.97 0.99"
    target_sparsity = "0.92 0.94 0.96 0.98 0.985 0.99"

    cmd = f"python /root/workspace/samsungstf/analize_code/1.1_optimal_operator_gain.py --ALGORITHM={{0}} --A_SPARSITY={{2}} --B_SPARSITY={{2}} --A_HEIGHT={target_size} --A_WIDTH={target_size} --B_HEIGHT={target_size} --B_WIDTH={target_size} --OUTPUT_FILE_PATH=/root/workspace/samsungstf/results/optimal_operator_gain_{target_size}_{{1}}_{{2}}.json"

    cmd_list = []
    temp_list = format_matrix_cmd(
        cmd, target_algorithms, target_algorithms.replace("_", "-")
    )
    temp_list = format_matrix_cmd(temp_list, target_sparsity)
    cmd_list += temp_list

    run_benchmark(cmd_list, 1, False)


def run_optimal_operator_profile():
    target_size = "1024"
    target_algorithms = "GEMM SpMMRowR SpMMRowC SpMMElemR SpMMElemC SpGEMMHashR SpGEMMHashC SpGEMMSortR SpGEMMSortC"
    target_sparsity_a = "0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.97 0.99"
    # target_sparsity_a = "0.4 0.6 0.8 0.9 0.95 0.97"
    target_sparsity_b = target_sparsity_a

    cmd = f"python /root/workspace/samsungstf/analize_code/1.1_optimal_operator_gain.py --ALGORITHM={{0}} --A_SPARSITY={{2}} --B_SPARSITY={{3}} --A_HEIGHT={target_size} --A_WIDTH={target_size} --B_HEIGHT={target_size} --B_WIDTH={target_size} --OUTPUT_FILE_PATH=/root/workspace/samsungstf/results/optimal_operator_profile_{target_size}_{{1}}_{{2}}_{{3}}.json"

    cmd_list = []
    temp_list = format_matrix_cmd(
        cmd, target_algorithms, target_algorithms.replace("_", "-")
    )
    temp_list = format_matrix_cmd(temp_list, target_sparsity_a)
    temp_list = format_matrix_cmd(temp_list, target_sparsity_b)
    cmd_list += temp_list

    run_benchmark(cmd_list, 1, False)


def run_period_sparsity_detection():
    cmd = f"python /root/workspace/samsungstf/analize_code/period_sparsity_detection.py --DEBUG_PRINT=False --N_TEST=100  --SPARSITY_LIST={{0}} --DETECT_PERIOD=32 --DETECT_PRESET={{2}} --OUTPUT_FILE_PATH=/root/workspace/samsungstf/results/period_sparsity_detection_method_{{1}}_{{2}}.json"
    sparsity_lists = glob_files(
        "/root/workspace/samsungstf/results", "matmul_history_.*\.json"
    )
    sparsity_lists_name = [Path(file).stem.replace("_", "-") for file in sparsity_lists]

    cmd_list = []
    temp_list = format_matrix_cmd(cmd, sparsity_lists, sparsity_lists_name)
    temp_list = format_matrix_cmd(temp_list, "Fixed Accurate Period Hyper-Predict")
    cmd_list += temp_list

    run_benchmark(cmd_list, 1, False)


def run_period_sparsity_detection_sensitivity():
    cmd = f"python /root/workspace/samsungstf/analize_code/period_sparsity_detection.py --DEBUG_PRINT=False --N_TEST=100 --USE_PREDICTION=True --SPARSITY_LIST={{0}} --DETECT_PERIOD={{2}} --DETECT_METHOD=sparsity_based --OUTPUT_FILE_PATH=/root/workspace/samsungstf/results/period_sparsity_detection_100sensitivity_{{1}}_{{2}}.json"
    sparsity_lists = glob_files(
        "/root/workspace/samsungstf/results", "matmul_history_.*\.json"
    )
    sparsity_lists_name = [Path(file).stem.replace("_", "-") for file in sparsity_lists]

    cmd_list = []
    temp_list = format_matrix_cmd(cmd, sparsity_lists, sparsity_lists_name)
    temp_list = format_matrix_cmd(temp_list, "-1 1 2 4 8 16 32")
    cmd_list += temp_list

    run_benchmark(cmd_list, 24, False)


def run_sparsity_variance():
    target_size = "1024"
    target_algorithms = "GEMM SpMMRowR SpMMRowC SpMMElemR SpMMElemC SpGEMMHashR SpGEMMHashC SpGEMMSortR SpGEMMSortC"
    target_sparsity = "0.2 0.4 0.6 0.7 0.8 0.9 0.95 0.97"
    # target_sparsity = "0.6 0.8 0.9"
    target_sparsity_std = "0 0.04 0.08 0.12 0.16 0.2"

    cmd = f"python /root/workspace/samsungstf/analize_code/2.1_variance_sensitivity.py --USE_PREDICTION=True --N_TEST=200 --ALGORITHM={{0}} --A_SPARSITY={{2}} --B_SPARSITY={{2}} --A_SPARSITY_STD={{3}} --A_HEIGHT={target_size} --A_WIDTH={target_size} --B_HEIGHT={target_size} --B_WIDTH={target_size} --OUTPUT_FILE_PATH=/root/workspace/samsungstf/results/variance_sensitivity_{target_size}_{{1}}_{{2}}_{{3}}.json"

    cmd_list = []
    temp_list = format_matrix_cmd(
        cmd, target_algorithms, target_algorithms.replace("_", "-")
    )
    temp_list = format_matrix_cmd(temp_list, target_sparsity)
    temp_list = format_matrix_cmd(temp_list, target_sparsity_std)
    cmd_list += temp_list

    run_benchmark(cmd_list, 24, False)


run_gpt()
