#ifndef TBFSMSTARPUCUDAOPTIONS_HPP
#define TBFSMSTARPUCUDAOPTIONS_HPP

#include <cstddef>
#include <stdexcept>
#include <string>

struct starpu_conf;

// Per-algorithm configuration. The caller owns runtime until construction ends.
struct TbfStarpuCudaOptions {
    enum class Aggregation { Single, M2L, P2P, Both };
    Aggregation aggregation = Aggregation::Single;
    std::size_t minTasks = 2;
    std::size_t m2lMaxTasks = 16;
    std::size_t p2pMaxTasks = 8;
    starpu_conf* runtime = nullptr;

    bool m2l() const { return aggregation == Aggregation::M2L || aggregation == Aggregation::Both; }
    bool p2p() const { return aggregation == Aggregation::P2P || aggregation == Aggregation::Both; }

    void validate() const {
        if(aggregation != Aggregation::Single && aggregation != Aggregation::M2L &&
           aggregation != Aggregation::P2P && aggregation != Aggregation::Both)
            throw std::invalid_argument("Invalid aggregation mode");
        if(minTasks < 2 || minTasks > 4)
            throw std::invalid_argument("Minimum group size must be between 2 and 4");
        if(m2lMaxTasks != 4 && m2lMaxTasks != 8 && m2lMaxTasks != 16)
            throw std::invalid_argument("Cross M2L maximum group size must be 4, 8, or 16");
        if(p2pMaxTasks != 4 && p2pMaxTasks != 8)
            throw std::invalid_argument("Cross P2P maximum group size must be 4 or 8");
#ifndef TBFMM_USE_GPU_COARSENING
        if(aggregation != Aggregation::Single)
            throw std::invalid_argument("GPU coarsening requires TBFMM_USE_GPU_COARSENING and the extended StarPU runtime");
#endif
    }
};

#endif
