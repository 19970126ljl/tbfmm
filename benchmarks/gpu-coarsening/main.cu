#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sched.h>
#include <starpu.h>
#include <starpu_coarsening.h>
#include "spacial/tbfmortonspaceindex.hpp"
#include "spacial/tbfspacialconfiguration.hpp"
#include "core/tbftree.hpp"
#include "kernels/rotationkernel/FRotationKernel.hpp"
#include "algorithms/sequential/tbfalgorithm.hpp"
#include "algorithms/smstarpu/tbfsmstarpualgorithmcuda.hpp"

using Real = double;
constexpr unsigned Order = 12;
constexpr size_t Coefficients = (Order + 2) * (Order + 1) / 2;
using Expansion = std::array<std::complex<Real>, Coefficients>;
using Tree = TbfTree<Real, Real, 4, Real, 4, Expansion, Expansion>;
// The same GPU assignment is used in every comparison mode.
struct Kernel : FRotationKernel<Real, Order> {
    using FRotationKernel<Real, Order>::FRotationKernel;
    static constexpr bool CpuM2L = false;
    static constexpr bool CpuP2P = false;
};
using GpuAlgorithm = TbfSmStarpuAlgorithmCuda<Real, Kernel>;
using CpuAlgorithm = TbfAlgorithm<Real, Kernel>;
using Clock = std::chrono::steady_clock;
using Aggregation = TbfStarpuCudaOptions::Aggregation;

struct Args {
    long particles = 1000000, height = 6, block = 32;
    int warmup = 1, repeats = 4, gpus = 1, cpuWorkers = 4, pipeline = 2, mainCpu = -1;
    unsigned seed = 0;
    size_t scan = 256;
    bool prepareModels = false;
    std::string aggregation, output, modelDirectory, distribution = "uniform";
    std::vector<unsigned> workerCpus;
    TbfStarpuCudaOptions options;
};

static long integer(const std::string& text) {
    size_t used = 0;
    const long value = std::stol(text, &used);
    if(used != text.size() || value < 0 || value > std::numeric_limits<int>::max())
        throw std::invalid_argument("Expected a non-negative integer: " + text);
    return value;
}

static Args parse(int argc, char** argv) {
    Args a;
    for(int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if(key == "--help") {
            std::cout << "Required: --aggregation single|m2l|p2p|both --output FILE "
                "--model-directory DIR --main-cpu CPU --worker-cpus GPU_DRIVERS,CPU_WORKERS\n"
                "Options: --particles 1000000 --height 6 --block 32 --seed 0 --distribution uniform|clustered "
                "--gpus 1 --cpu-workers 4 --warmup 1 --repeats 4 --pipeline 2 --scan 256 --min-tasks 2 "
                "--m2l-max 16 --p2p-max 8 --prepare-models 0|1\n";
            std::exit(0);
        }
        if(++i == argc) throw std::invalid_argument("Missing value for " + key);
        const std::string value = argv[i];
        if(key == "--aggregation") a.aggregation = value;
        else if(key == "--output") a.output = value;
        else if(key == "--model-directory") a.modelDirectory = value;
        else if(key == "--distribution") a.distribution = value;
        else if(key == "--worker-cpus") {
            std::istringstream stream(value);
            std::string cpu;
            while(std::getline(stream, cpu, ',')) a.workerCpus.push_back(integer(cpu));
            if(value.empty() || value.back() == ',') throw std::invalid_argument("Invalid worker CPU list");
        }
        else if(key == "--particles") a.particles = integer(value);
        else if(key == "--height") a.height = integer(value);
        else if(key == "--block") a.block = integer(value);
        else if(key == "--seed") a.seed = integer(value);
        else if(key == "--gpus") a.gpus = integer(value);
        else if(key == "--cpu-workers") a.cpuWorkers = integer(value);
        else if(key == "--pipeline") a.pipeline = integer(value);
        else if(key == "--main-cpu") a.mainCpu = integer(value);
        else if(key == "--warmup") a.warmup = integer(value);
        else if(key == "--repeats") a.repeats = integer(value);
        else if(key == "--scan") a.scan = integer(value);
        else if(key == "--min-tasks") a.options.minTasks = integer(value);
        else if(key == "--m2l-max") a.options.m2lMaxTasks = integer(value);
        else if(key == "--p2p-max") a.options.p2pMaxTasks = integer(value);
        else if(key == "--prepare-models") {
            if(value != "0" && value != "1") throw std::invalid_argument("prepare-models must be 0 or 1");
            a.prepareModels = value == "1";
        }
        else throw std::invalid_argument("Unknown option: " + key);
    }
    if(a.aggregation == "single") a.options.aggregation = Aggregation::Single;
    else if(a.aggregation == "m2l") a.options.aggregation = Aggregation::M2L;
    else if(a.aggregation == "p2p") a.options.aggregation = Aggregation::P2P;
    else if(a.aggregation == "both") a.options.aggregation = Aggregation::Both;
    else throw std::invalid_argument("Specify --aggregation single|m2l|p2p|both");
    a.options.validate();
    if(a.output.empty() || a.modelDirectory.empty() || a.mainCpu < 0 || a.workerCpus.empty())
        throw std::invalid_argument("Output, model directory, main CPU and worker CPUs are required");
    if(a.particles < 1 || a.height < 3 || a.height > 20 || a.block < 1 || a.repeats < 1 ||
       a.gpus < 1 || a.cpuWorkers < 1 || a.pipeline > 2 || a.scan < 1 ||
       a.workerCpus.size() != size_t(a.gpus + a.cpuWorkers) || a.workerCpus.size() > STARPU_NMAXWORKERS ||
       a.warmup > std::numeric_limits<int>::max() - a.repeats)
        throw std::invalid_argument("Unsupported input or worker configuration; pipeline must be 0, 1, or 2");
    if(a.distribution != "uniform" && a.distribution != "clustered")
        throw std::invalid_argument("Distribution must be uniform or clustered");
    std::set<unsigned> cpus(a.workerCpus.begin(), a.workerCpus.end());
    if(cpus.size() != a.workerCpus.size() || !cpus.insert(a.mainCpu).second)
        throw std::invalid_argument("Main thread and workers require distinct CPUs");
    cpu_set_t allowed;
    if(sched_getaffinity(0, sizeof(allowed), &allowed)) throw std::runtime_error("Cannot read CPU allocation");
    for(auto cpu : cpus) if(cpu >= CPU_SETSIZE || !CPU_ISSET(cpu, &allowed))
        throw std::invalid_argument("Requested CPU is outside the process allocation");
    return a;
}

extern char** environ;
static void configureRuntime(Args& args, starpu_conf& conf) {
    // CLI and the documented preset determine settings, never inherited STARPU_*.
    std::vector<std::string> names;
    for(char** p = environ; *p; ++p) {
        const std::string item = *p;
        if(item.rfind("STARPU_", 0) == 0) names.push_back(item.substr(0, item.find('=')));
    }
    for(const auto& name : names) unsetenv(name.c_str());
    const auto set = [](const char* key, const std::string& value) {
        if(setenv(key, value.c_str(), 1)) throw std::runtime_error("Cannot configure " + std::string(key));
    };
    set("STARPU_CUDA_PIPELINE", std::to_string(args.pipeline));
    set("STARPU_NWORKER_PER_CUDA", "1");
    set("STARPU_CUDA_THREAD_PER_DEV", "1");
    set("STARPU_CUDA_THREAD_PER_WORKER", "0");
    set("STARPU_WORKERS_NOBIND", args.prepareModels ? "1" : "0");
    set("STARPU_MAIN_THREAD_BIND", "0");
    set("STARPU_MAIN_THREAD_CPUID", std::to_string(args.mainCpu));
    set("STARPU_HOME", args.modelDirectory);
    set("STARPU_PERF_MODEL_DIR", args.modelDirectory);
    set("STARPU_PREFETCH", "1");
    set("STARPU_PROFILING", "0");
    set("STARPU_CODELET_PROFILING", "0");
    set("STARPU_FXT_TRACE", "0");
    set("STARPU_SILENT", "1");
    starpu_coarsening_options grouping{};
    grouping.enabled = 1;
    grouping.scan_limit = args.scan;
    grouping.max_tasks = 16;
    if(starpu_coarsening_configure(nullptr, &grouping)) throw std::runtime_error("Cannot configure GPU coarsening");
    starpu_conf_init(&conf);
    conf.precedence_over_environment_variables = 1;
    conf.sched_policy_name = "prio";
    conf.ncpus = args.cpuWorkers;
    conf.ncuda = args.gpus;
    conf.nhip = conf.nsycl = conf.nopencl = conf.nmax_fpga = conf.nmpi_sc = conf.ntcpip_sc = 0;
    conf.calibrate = 0;
    conf.bus_calibrate = args.prepareModels ? 1 : 0;
    conf.use_explicit_workers_bindid = 1;
    conf.use_explicit_workers_cuda_gpuid = 1;
    for(size_t i = 0; i < args.workerCpus.size(); ++i) conf.workers_bindid[i] = args.workerCpus[i];
    for(int i = 0; i < args.gpus; ++i) conf.workers_cuda_gpuid[i] = i;
    cpu_set_t main;
    CPU_ZERO(&main);
    CPU_SET(args.mainCpu, &main);
    if(sched_setaffinity(0, sizeof(main), &main)) throw std::runtime_error("Cannot bind the main thread");
    args.options.runtime = &conf;
}

#ifdef TBFMM_VALIDATE
static std::vector<std::array<Real, 4>> collect(const Tree& tree, size_t count) {
    std::vector<std::array<Real, 4>> result(count);
    std::vector<unsigned> seen(count);
    tree.applyToAllLeaves([&](const auto& header, const long* indexes, const auto&, const auto& rhs) {
        for(long i = 0; i < header.nbParticles; ++i) {
            if(indexes[i] < 0 || size_t(indexes[i]) >= count || seen[indexes[i]]++)
                throw std::runtime_error("Particle identity mismatch");
            for(int c = 0; c < 4; ++c) result[indexes[i]][c] = rhs[c][i];
        }
    });
    for(auto n : seen) if(n != 1) throw std::runtime_error("Missing particle");
    return result;
}

static std::array<double, 4> compare(const std::vector<std::array<Real, 4>>& reference,
                                   const std::vector<std::array<Real, 4>>& result) {
    std::array<double, 4> error{};
    for(int c = 0; c < 4; ++c) {
        long double sum = 0, norm = 0;
        for(size_t i = 0; i < result.size(); ++i) {
            if(!std::isfinite(result[i][c])) throw std::runtime_error("Non-finite output");
            const long double delta = result[i][c] - reference[i][c];
            sum += delta * delta;
            norm += static_cast<long double>(reference[i][c]) * reference[i][c];
        }
        error[c] = std::sqrt(double(sum / std::max(norm, 1e-300L)));
        if(error[c] > 1e-11) throw std::runtime_error("CPU reference relative L2 exceeds 1e-11");
    }
    return error;
}
#endif

int main(int argc, char** argv) {
    try {
        auto args = parse(argc, argv);
        std::ofstream out(args.output);
        if(!out) throw std::runtime_error("Cannot open output");
        std::mt19937 rng(args.seed);
        std::uniform_real_distribution<Real> uniform(0, 1);
        std::normal_distribution<Real> normal(0, 0.065);
        std::vector<std::array<Real, 4>> particles(args.particles);
        for(long i = 0; i < args.particles; ++i) {
            for(int c = 0; c < 3; ++c) {
                Real x;
                if(args.distribution == "uniform") x = uniform(rng);
                else do { x = (i % 2 ? 0.72 : 0.28) + normal(rng); } while(x <= 0 || x >= 1);
                particles[i][c] = x;
            }
            particles[i][3] = 0.1;
        }
        uint64_t inputHash = 1469598103934665603ULL;
        const auto* bytes = reinterpret_cast<const unsigned char*>(particles.data());
        for(size_t i = 0; i < particles.size() * sizeof(particles[0]); ++i) {
            inputHash ^= bytes[i]; inputHash *= 1099511628211ULL;
        }
        TbfSpacialConfiguration<Real, 3> config(args.height, std::array<Real, 3>{1, 1, 1}, std::array<Real, 3>{0.5, 0.5, 0.5});
#ifdef TBFMM_VALIDATE
        Tree referenceTree(config, particles, args.block, true);
        CpuAlgorithm referenceAlgorithm(config);
        referenceAlgorithm.execute(referenceTree);
        const auto reference = collect(referenceTree, particles.size());
#endif
        starpu_conf runtime;
        configureRuntime(args, runtime);
        if(args.prepareModels) {
            // Bus-only startup: calibration threads inherit the bound main CPU.
            runtime.ncpus = 0;
            runtime.use_explicit_workers_bindid = 0;
            const int status = starpu_init(&runtime);
            if(status) throw std::runtime_error("Bus model preparation failed");
            starpu_shutdown();
            out << "{\"purpose\":\"model-preparation\",\"pass\":true}\n";
            return 0;
        }
        auto algorithm = std::make_unique<GpuAlgorithm>(config, TbfDefaultLastLevel, args.options);
        if(sched_getcpu() != args.mainCpu) throw std::runtime_error("Main thread CPU binding changed");
        if(starpu_cuda_worker_get_count() != unsigned(args.gpus) || starpu_cpu_worker_get_count() != unsigned(args.cpuWorkers))
            throw std::runtime_error("Actual worker counts differ from the requested counts");
        for(unsigned w = 0; w < starpu_worker_get_count(); ++w)
            if(starpu_worker_get_bindid(w) != int(args.workerCpus[w]))
                throw std::runtime_error("Actual worker CPU binding differs from the requested binding");
        out << std::setprecision(17) << "{\"purpose\":\""
#ifdef TBFMM_VALIDATE
            << "validation"
#else
            << "performance"
#endif
            << "\",\"particles\":" << args.particles << ",\"height\":" << args.height << ",\"block\":" << args.block
            << ",\"order\":" << Order << ",\"coefficients\":" << Coefficients << ",\"input_hash\":" << inputHash
            << ",\"distribution\":\"" << args.distribution << "\",\"seed\":" << args.seed
            << ",\"aggregation\":\"" << args.aggregation << "\",\"gpu_operators\":[\"m2l\",\"p2p\"],\"scheduler\":\"prio\""
            << ",\"pipeline\":" << args.pipeline << ",\"scan\":" << args.scan << ",\"min_tasks\":" << args.options.minTasks
            << ",\"m2l_max\":" << args.options.m2lMaxTasks << ",\"p2p_max\":" << args.options.p2pMaxTasks
            << ",\"inside_max\":4,\"prepare_models\":" << (args.prepareModels ? "true" : "false") << ",\"workers\":[";
        for(unsigned w = 0; w < starpu_worker_get_count(); ++w) {
            if(w) out << ',';
            out << "{\"id\":" << w << ",\"type\":" << starpu_worker_get_type(w) << ",\"device\":" << starpu_worker_get_devid(w)
                << ",\"bind_cpu\":" << starpu_worker_get_bindid(w) << '}';
        }
        out << "],\"iterations\":[";
        for(int it = 0; it < args.warmup + args.repeats; ++it) {
            const auto buildBegin = Clock::now();
            Tree tree(config, particles, args.block, true);
            const auto buildEnd = Clock::now();
            const auto begin = Clock::now();
            algorithm->execute(tree);
            const double elapsed = std::chrono::duration<double>(Clock::now() - begin).count();
            if(it) out << ',';
            out << "{\"iteration\":" << it << ",\"warmup\":" << (it < args.warmup ? "true" : "false")
                << ",\"execute_seconds\":" << elapsed << ",\"tree_build_seconds\":"
                << std::chrono::duration<double>(buildEnd - buildBegin).count();
#ifdef TBFMM_VALIDATE
            const auto error = compare(reference, collect(tree, particles.size()));
            out << ",\"cpu_reference_relative_l2\":[";
            for(int c = 0; c < 4; ++c) { if(c) out << ','; out << error[c]; }
            out << ']';
#endif
            out << '}'; out.flush();
            std::cout << "Iteration " << it << ": execute " << elapsed << " s\n" << std::flush;
        }
        algorithm.reset();
        starpu_coarsening_stats totals{};
        for(unsigned w = 0; w < STARPU_NMAXWORKERS; ++w) {
            starpu_coarsening_stats local{};
            if(starpu_coarsening_get_stats(w, &local) == 0) {
                totals.groups += local.groups; totals.members += local.members;
                totals.cancelled += local.cancelled; totals.rejected += local.rejected;
            }
        }
        if(args.aggregation == "single" && (totals.groups || totals.members))
            throw std::runtime_error("Single mode unexpectedly executed a physical group");
        out << "],\"groups\":" << totals.groups << ",\"grouped_members\":" << totals.members
            << ",\"cancelled\":" << totals.cancelled << ",\"rejected\":" << totals.rejected << ",\"pass\":true}\n";
        return 0;
    }
    catch(const std::exception& error) { std::cerr << "Error: " << error.what() << '\n'; return 1; }
}
