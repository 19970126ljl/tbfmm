if(CMAKE_VERSION VERSION_LESS 3.18)
    message(FATAL_ERROR "GPU coarsening requires CMake 3.18 or newer")
endif()
if(NOT "STARPU" IN_LIST TBF_USE_LIST OR NOT "CUDA" IN_LIST TBF_USE_LIST)
    message(FATAL_ERROR "GPU coarsening requires TBFMM_USE_STARPU=<prefix>, TBFMM_ENABLE_STARPU=ON and TBFMM_USE_CUDA=ON")
endif()

find_library(TBFMM_COARSENING_STARPU_LIBRARY
    NAMES starpu-${TBFMM_STARPU_VERSION}
    PATHS "${TBFMM_USE_STARPU}/lib" "${TBFMM_USE_STARPU}/lib64"
    NO_DEFAULT_PATH REQUIRED)

include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES ${TBFMM_STARPU_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set(CMAKE_REQUIRED_LIBRARIES ${TBFMM_COARSENING_STARPU_LIBRARY})
check_cxx_source_compiles("
#include <starpu.h>
#include <starpu_coarsening.h>
int main() {
    starpu_codelet cl{};
    starpu_coarsened_variant variant{};
    cl.coarsened_variant = &variant;
    starpu_coarsening_options options{1, 256, 16};
    return starpu_coarsening_configure(nullptr, &options);
}" TBFMM_COARSENING_ABI_AVAILABLE)
unset(CMAKE_REQUIRED_INCLUDES)
unset(CMAKE_REQUIRED_LIBRARIES)
if(NOT TBFMM_COARSENING_ABI_AVAILABLE)
    message(FATAL_ERROR "The selected StarPU headers/library do not provide the coarsened_variant ABI and starpu_coarsening_configure; select a matching task-coarsening runtime build")
endif()

foreach(target IN ITEMS gpu-coarsening-benchmark gpu-coarsening-validate)
    add_executable(${target} main.cu)
    target_include_directories(${target} BEFORE PRIVATE ${TBFMM_STARPU_INCLUDE_DIRS})
    target_compile_definitions(${target} PRIVATE TBFMM_USE_GPU_COARSENING)
    target_link_libraries(${target} PRIVATE ${TBFMM_COARSENING_STARPU_LIBRARY} ${TBFMM_LIBRARIES})
    set_target_properties(${target} PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
endforeach()
target_compile_definitions(gpu-coarsening-validate PRIVATE TBFMM_VALIDATE)
target_compile_options(gpu-coarsening-validate PRIVATE -UNDEBUG)
