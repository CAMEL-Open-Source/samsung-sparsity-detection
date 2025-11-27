# 요구 CMake 최소 버전
cmake_minimum_required(VERSION 3.8)

# 프로젝트 이름
project("io_analyses" CXX CUDA)
set(CMAKE_VERBOSE_MAKEFILE true)
set(CMAKE_CXX_STANDARD 17)

# Add options for sanitizer
option(USE_SANITIZER "Use sanitizer." OFF)

if(USE_SANITIZER)
    set(SANFLAG
        -fsanitize=address
        -fsanitize=undefined
        -fno-sanitize-recover=all
        -fsanitize=float-divide-by-zero
        -fsanitize=float-cast-overflow
        -fno-sanitize=null
        -fno-sanitize=alignment
    )
else()
    set(SANFLAG "")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)

# 빌드 대상 바이너리 추가
add_executable(count_nonzero count_nonzero.cu)
add_executable(gemm_ntile gemm_ntile.cu)
add_executable(gemm_tile gemm_tile.cu)
add_executable(spmm_cusparse spmm_cusparse.cu)
add_executable(spmm_csr_row spmm_csr_row.cu)
add_executable(spmm_csr_col spmm_csr_col.cu)
add_executable(spmm_csr_elem_row spmm_csr_elem_row.cu)
add_executable(spmm_csr_elem_col spmm_csr_elem_col.cu)
add_executable(spgemm_cusparse spgemm_cusparse.cu)
add_executable(spgemm_sparse_hash_row spgemm_sparse_hash_row.cu)
add_executable(spgemm_sparse_hash_col spgemm_sparse_hash_col.cu)
add_executable(spgemm_sparse_sort_row spgemm_sparse_sort_row.cu)
add_executable(spgemm_sparse_sort_col spgemm_sparse_sort_col.cu)
add_executable(bincount bincount.cu)
add_executable(bincount_random bincount_random.cu)
add_executable(1_embedding_continues 1_embedding_continues.cu)
add_executable(2_embedding_copy 2_embedding_copy.cu)
add_executable(3_table_change 3_table_change.cu)
add_executable(4_calculate 4_calculate.cu)

target_link_libraries(count_nonzero
    PUBLIC cudart cusparse nvToolsExt
)
target_link_libraries(gemm_ntile
    PUBLIC cudart cusparse nvToolsExt
)
target_link_libraries(gemm_tile
    PUBLIC cudart cusparse nvToolsExt
)
target_link_libraries(spmm_cusparse
    PUBLIC cudart cusparse nvToolsExt
)
target_link_libraries(spmm_csr_row
    PUBLIC cudart cusparse nvToolsExt
)
target_link_libraries(spmm_csr_col
    PUBLIC cudart cusparse nvToolsExt
)
target_link_libraries(spmm_csr_elem_row
    PUBLIC cudart cusparse nvToolsExt
)
target_link_libraries(spmm_csr_elem_col
    PUBLIC cudart cusparse nvToolsExt
)
target_link_libraries(spgemm_cusparse
    PUBLIC cudart cusparse nvToolsExt
)
target_link_libraries(spgemm_sparse_hash_row
    PUBLIC cudart cusparse nvToolsExt
)
target_link_libraries(spgemm_sparse_hash_col
    PUBLIC cudart cusparse nvToolsExt
)
target_link_libraries(spgemm_sparse_sort_row
    PUBLIC cudart cusparse nvToolsExt
)
target_link_libraries(spgemm_sparse_sort_col
    PUBLIC cudart cusparse nvToolsExt
)
target_link_libraries(bincount
    PUBLIC cudart cusparse nvToolsExt
)
target_link_libraries(bincount_random
    PUBLIC cudart cusparse nvToolsExt
)
