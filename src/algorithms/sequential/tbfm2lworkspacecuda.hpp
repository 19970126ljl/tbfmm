#ifndef TBFM2LWORKSPACECUDA_HPP
#define TBFM2LWORKSPACECUDA_HPP

// Included inside TbfGroupKernelInterfaceCuda_core, after M2LInteraction.
// Host-only worker-local storage. All tasks use the runtime's local stream;
// events protect both host upload staging and device metadata across returns.
struct M2LWorkspace {
    struct Slot {
        M2LInteraction* hostInteractions = nullptr;
        M2LInteraction* deviceInteractions = nullptr;
        long int* hostOffsets = nullptr;
        long int* deviceOffsets = nullptr;
        size_t interactionCapacity = 0, offsetCapacity = 0;
        cudaEvent_t done = nullptr;
        bool busy = false;
        std::vector<M2LInteraction> interactions;
        std::vector<long int> offsets;

        void wait(){
            if(busy){
                const auto status = cudaEventQuery(done);
                if(status == cudaErrorNotReady){ CUDA_ASSERT(cudaEventSynchronize(done)); }
                else { CUDA_ASSERT(status); }
                busy = false;
            }
        }
        template<class T>
        static void reserve(T*& host, T*& device, size_t& capacity, const size_t requested){
            if(requested <= capacity) return;
            size_t grown = std::max(size_t(64), capacity);
            while(grown < requested) grown *= 2;
            // The caller waited for this slot before replacing either buffer.
            if(host){ CUDA_ASSERT(cudaFreeHost(host)); }
            if(device){ CUDA_ASSERT(cudaFree(device)); }
            CUDA_ASSERT(cudaMallocHost(&host, grown*sizeof(T)));
            CUDA_ASSERT(cudaMalloc(&device, grown*sizeof(T)));
            capacity = grown;
        }
        void upload(cudaStream_t stream){
            reserve(hostInteractions, deviceInteractions, interactionCapacity, interactions.size());
            reserve(hostOffsets, deviceOffsets, offsetCapacity, offsets.size());
            std::copy(interactions.begin(), interactions.end(), hostInteractions);
            std::copy(offsets.begin(), offsets.end(), hostOffsets);
            CUDA_ASSERT(cudaMemcpyAsync(deviceInteractions, hostInteractions,
                interactions.size()*sizeof(M2LInteraction), cudaMemcpyHostToDevice, stream));
            CUDA_ASSERT(cudaMemcpyAsync(deviceOffsets, hostOffsets,
                offsets.size()*sizeof(long int), cudaMemcpyHostToDevice, stream));
        }
        void record(cudaStream_t stream){
            if(!done){ CUDA_ASSERT(cudaEventCreateWithFlags(&done, cudaEventDisableTiming)); }
            CUDA_ASSERT(cudaEventRecord(done, stream));
            busy = true;
        }
        void release(){
            wait();
            if(done){ CUDA_ASSERT(cudaEventDestroy(done)); done = nullptr; }
            if(hostInteractions){ CUDA_ASSERT(cudaFreeHost(hostInteractions)); hostInteractions = nullptr; }
            if(hostOffsets){ CUDA_ASSERT(cudaFreeHost(hostOffsets)); hostOffsets = nullptr; }
            if(deviceInteractions){ CUDA_ASSERT(cudaFree(deviceInteractions)); deviceInteractions = nullptr; }
            if(deviceOffsets){ CUDA_ASSERT(cudaFree(deviceOffsets)); deviceOffsets = nullptr; }
            interactionCapacity = offsetCapacity = 0;
            interactions.clear(); offsets.clear();
        }
    };
    std::array<Slot,2> slots;
    size_t next = 0;
    Slot& acquire(){
        auto& slot = slots[next++ % slots.size()];
        slot.wait();
        slot.interactions.clear(); slot.offsets.clear();
        return slot;
    }
    void release(){for(auto& slot : slots) slot.release(); next = 0;}
};

inline M2LWorkspace& GetM2LWorkspace(){
    // Explicitly release on the CUDA worker before StarPU tears its device
    // down. The TLS destructor itself performs no CUDA calls.
    static thread_local M2LWorkspace workspace;
    return workspace;
}

#endif
