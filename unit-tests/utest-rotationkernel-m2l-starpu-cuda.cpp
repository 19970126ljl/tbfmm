// See LICENSE file at project root.
// @TBF_USE_STARPU
// @TBF_USE_CUDA

#include "spacial/tbfmortonspaceindex.hpp"
#include "core/tbftree.hpp"
#include "kernels/rotationkernel/FRotationKernel.hpp"
#include "algorithms/sequential/tbfalgorithm.hpp"
#include "algorithms/smstarpu/tbfsmstarpualgorithmcuda.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>

// CPU workers run all other stages. A successful run therefore proves M2L
// executed on CUDA, even when StarPU would prefer the CPU for small tasks.
template <class Real, int Order>
struct GpuM2LOnly : FRotationKernel<Real, Order> {
    using FRotationKernel<Real, Order>::FRotationKernel;
    static constexpr bool CpuM2L = false;
    static constexpr bool CudaP2P = false;
};

static void Require(const bool condition, const char* message){
    if(!condition) throw std::runtime_error(message);
}

struct Error {
    double squared = 0;
    double referenceSquared = 0;
    double largest = 0;
    double referenceLargest = 0;

    template <class Value>
    void add(const Value reference, const Value result){
        const double difference = std::abs(reference-result);
        const double magnitude = std::abs(reference);
        Require(std::isfinite(difference) && std::isfinite(magnitude), "non-finite result");
        squared += difference*difference;
        referenceSquared += magnitude*magnitude;
        largest = std::max(largest, difference);
        referenceLargest = std::max(referenceLargest, magnitude);
    }
    void check(const double tolerance, const char* name, const bool verbose = true) const {
        const double relative = std::sqrt(squared/std::max(referenceSquared, 1e-300));
        if(verbose || relative > tolerance){
            std::cout << name << ": relative L2=" << relative << ", max error=" << largest << '\n';
        }
        Require(relative <= tolerance && largest <= tolerance*std::max(referenceLargest, 1e-30),
                "CPU/CUDA comparison failed");
    }
};

template <class Tree>
void CompareCells(const Tree& reference, const Tree& result, const double tolerance){
    Error multipoles, locals;
    using Expansion = typename Tree::CellGroupClass::LocalClass;
    std::array<Error,std::tuple_size<Expansion>::value> localsByCoefficient{};
    for(long int level = 0; level < reference.getSpacialConfiguration().getTreeHeight(); ++level){
        const auto& refGroups = reference.getCellGroupsAtLevel(level);
        const auto& groups = result.getCellGroupsAtLevel(level);
        Require(refGroups.size() == groups.size(), "different cell group counts");
        for(std::size_t group = 0; group < groups.size(); ++group){
            for(long int cell = 0; cell < groups[group].getNbCells(); ++cell){
                Require(refGroups[group].getCellSpacialIndex(cell) == groups[group].getCellSpacialIndex(cell),
                        "different cell indexes");
                const auto& refM = refGroups[group].getCellMultipole(cell);
                const auto& refL = refGroups[group].getCellLocal(cell);
                const auto& m = groups[group].getCellMultipole(cell);
                const auto& l = groups[group].getCellLocal(cell);
                for(std::size_t idx = 0; idx < m.size(); ++idx){
                    multipoles.add(refM[idx], m[idx]);
                    locals.add(refL[idx], l[idx]);
                    localsByCoefficient[idx].add(refL[idx], l[idx]);
                }
            }
        }
    }
    multipoles.check(tolerance, "M");
    locals.check(tolerance, "L");
    // Different degrees have very different scales; a large coefficient must
    // not hide an error in a smaller coefficient in the overall norm.
    for(const auto& error : localsByCoefficient) error.check(tolerance, "L coefficient", false);
}

template <class Tree>
void CompareParticles(const Tree& reference, const Tree& result, const double tolerance){
    std::array<Error,4> errors;
    const auto& refGroups = reference.getParticleGroups();
    const auto& groups = result.getParticleGroups();
    for(std::size_t group = 0; group < groups.size(); ++group){
        for(long int leaf = 0; leaf < groups[group].getNbLeaves(); ++leaf){
            const auto refRhs = refGroups[group].getParticleRhs(leaf);
            const auto rhs = groups[group].getParticleRhs(leaf);
            for(long int particle = 0; particle < groups[group].getNbParticlesInLeaf(leaf); ++particle){
                for(int value = 0; value < 4; ++value) errors[value].add(refRhs[value][particle], rhs[value][particle]);
            }
        }
    }
    for(const auto& error : errors) error.check(tolerance, "particle RHS");
}

template <class Real, class Tree>
void SeedExpansions(Tree& tree){
    tree.applyToAllCells([](const long int level, const auto& symbolic, auto multipole, auto local){
        for(std::size_t idx = 0; idx < multipole->get().size(); ++idx){
            // Scaled coefficients prevent high-order/level translations from
            // making the reference dominated by a few enormous components.
            const Real scale = std::pow(Real(0.03), Real(idx+1));
            multipole->get()[idx] = {scale*Real(1+symbolic.spaceIndex%7), scale*Real(idx%3)};
            local->get()[idx] = {Real(0.01)*(level+1)/(idx+1), Real(-0.005)/(idx+1)};
        }
    });
}

template <class Real, int Order>
void RunSuite(const long int height, const bool quick = false){
    using CpuKernel = FRotationKernel<Real, Order>;
    using GpuKernel = GpuM2LOnly<Real, Order>;
    using Expansion = std::array<std::complex<Real>, (Order+1)*(Order+2)/2>;
    using Tree = TbfTree<Real, Real, 4, Real, 4, Expansion, Expansion>;
    const TbfSpacialConfiguration<Real,3> configuration(height, {{1,1,1}}, {{Real(0.5),Real(0.5),Real(0.5)}});
    auto cpu = std::make_unique<TbfAlgorithm<Real, CpuKernel>>(configuration);
    TbfSmStarpuAlgorithmCuda<Real, GpuKernel> gpu(configuration);
    Require(starpu_cuda_worker_get_count() > 0, "this test requires a CUDA worker");
    static_assert(!TbfAlgorithmUtils::KernelHardwareSupport<GpuKernel>::CpuM2L);
    static_assert(TbfAlgorithmUtils::KernelHardwareSupport<GpuKernel>::CudaM2L);
    starpu_profiling_status_set(STARPU_PROFILING_ENABLE);
    const double tolerance = std::is_same<Real,float>::value ? 4e-5 : 2e-12;

    const std::vector<long int> groupSizes = quick ? std::vector<long int>{7,4096} : std::vector<long int>{1,7,4096};
    for(const long int groupSize : groupSizes){
        for(const bool clustered : {false, true}){
            std::mt19937 generator(1729);
            std::uniform_real_distribution<double> uniform(0.01,0.99);
            std::vector<std::array<Real,4>> particles(quick ? 33 : 257);
            for(std::size_t idx = 0; idx < particles.size(); ++idx){
                for(int dim = 0; dim < 3; ++dim){
                    const double position = uniform(generator);
                    particles[idx][dim] = Real(clustered ? (idx%2 ? 0.04+0.12*position : 0.64+0.30*position) : position);
                }
                particles[idx][3] = Real(0.01+0.001*(idx%7));
            }
            std::cout << "order=" << Order << " height=" << height << " group=" << groupSize
                      << " clustered=" << clustered << std::endl;
            Tree ref(configuration, particles, groupSize, false);
            Tree result(configuration, particles, groupSize, false);
            SeedExpansions<Real>(ref);
            SeedExpansions<Real>(result);
            // Two executions check both nonzero initial L and repeat lifetimes.
            for(int repeat = 0; repeat < 2; ++repeat){
                cpu->execute(ref, TbfAlgorithmUtils::TbfM2L);
                gpu.execute(result, TbfAlgorithmUtils::TbfM2L);
                CompareCells(ref, result, tolerance);
            }
            Tree fullRef(configuration, particles, groupSize, false);
            Tree fullResult(configuration, particles, groupSize, false);
            cpu->execute(fullRef);
            gpu.execute(fullResult);
            CompareCells(fullRef, fullResult, tolerance);
            CompareParticles(fullRef, fullResult, tolerance);
        }
    }

    int executedCudaTasks = 0;
    for(int worker = 0; worker < static_cast<int>(starpu_worker_get_count()); ++worker){
        if(starpu_worker_get_type(worker) == STARPU_CUDA_WORKER){
            starpu_profiling_worker_info info{};
            Require(starpu_profiling_worker_get_info(worker, &info) == 0, "worker profiling query failed");
            std::cout << "CUDA device " << starpu_worker_get_devid(worker) << " M2L tasks=" << info.executed_tasks << '\n';
            executedCudaTasks += info.executed_tasks;
        }
    }
    // StarPU may legitimately leave a device idle (for example with DMDA on
    // heterogeneous GPUs). Check actual CUDA execution without prescribing
    // the scheduler's per-device placement.
    Require(executedCudaTasks > 0, "no M2L tasks executed on CUDA");
}

int main(int argc, char** argv){
    try{
        const bool quick = argc == 2 && std::string(argv[1]) == "--quick";
        Require(argc == 1 || quick, "usage: utest-rotationkernel-m2l-starpu-cuda [--quick]");
        RunSuite<double,12>(quick ? 3 : 4, quick);
        RunSuite<double,5>(3, quick);
        RunSuite<float,5>(3, quick);
        // 153 coefficients exercise multiple coefficients per thread.
        RunSuite<double,16>(3, true);
        std::cout << "StarPU CUDA M2L checks passed" << std::endl;
        return 0;
    }
    catch(const std::exception& error){
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
