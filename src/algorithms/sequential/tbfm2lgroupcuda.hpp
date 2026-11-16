#ifndef TBFM2LGROUPCUDA_HPP
#define TBFM2LGROUPCUDA_HPP
// Included in TbfGroupKernelInterfaceCuda_core after M2L_core.
struct M2LGroupDescriptor {
    long int level, interaction, offset, targets;
    std::array<std::pair<unsigned char*,size_t>,3> target;
    std::array<std::pair<const unsigned char*,size_t>,3> source;
};
// A bounded small group is copied into CUDA kernel parameters at launch.
// Only member metadata is passed here; mathematical interaction contents are
// rebuilt and uploaded for every physical execution.
struct M2LCompactArguments {
    M2LGroupDescriptor members[4];
    long int begin[4];
    long int targets=0;
};
template<size_t Capacity>struct M2LWideArguments {
    M2LGroupDescriptor members[Capacity];
    long int begin[Capacity]{};
    long int targets=0;
};
struct M2LGroupSlot {
    M2LInteraction *hostInteractions=nullptr,*deviceInteractions=nullptr;
    long int *hostOffsets=nullptr,*deviceOffsets=nullptr;
    size_t interactionsCapacity=0,offsetsCapacity=0;
    std::vector<M2LGroupDescriptor> descriptors;
    std::vector<M2LInteraction> interactions,temporary;
    std::vector<long int> offsets;
    M2LCompactArguments compact{};
    M2LWideArguments<8> wide8{};
    M2LWideArguments<16> wide16{};
    template<size_t Capacity>void uploadWide(M2LWideArguments<Capacity>& args,cudaStream_t stream){

        if(descriptors.size()>Capacity)throw std::runtime_error("M2L wide capacity exceeded");
        args={};
        for(size_t i=0;i<descriptors.size();++i){args.members[i]=descriptors[i];args.begin[i]=args.targets;args.targets+=descriptors[i].targets;}
        M2LWorkspace::Slot::reserve(hostInteractions,deviceInteractions,interactionsCapacity,interactions.size());
        M2LWorkspace::Slot::reserve(hostOffsets,deviceOffsets,offsetsCapacity,offsets.size());
        std::copy(interactions.begin(),interactions.end(),hostInteractions);std::copy(offsets.begin(),offsets.end(),hostOffsets);
        CUDA_ASSERT(cudaMemcpyAsync(deviceInteractions,hostInteractions,interactions.size()*sizeof(M2LInteraction),cudaMemcpyHostToDevice,stream));
        CUDA_ASSERT(cudaMemcpyAsync(deviceOffsets,hostOffsets,offsets.size()*sizeof(long int),cudaMemcpyHostToDevice,stream));
    }
    void uploadCompact(cudaStream_t stream){
        compact.targets=0;
        assert(descriptors.size()<=4);
        for(size_t i=0;i<descriptors.size();++i){
            compact.members[i]=descriptors[i];compact.begin[i]=compact.targets;
            compact.targets+=descriptors[i].targets;
        }
        M2LWorkspace::Slot::reserve(hostInteractions,deviceInteractions,interactionsCapacity,interactions.size());
        M2LWorkspace::Slot::reserve(hostOffsets,deviceOffsets,offsetsCapacity,offsets.size());
        std::copy(interactions.begin(),interactions.end(),hostInteractions);
        std::copy(offsets.begin(),offsets.end(),hostOffsets);
        CUDA_ASSERT(cudaMemcpyAsync(deviceInteractions,hostInteractions,interactions.size()*sizeof(M2LInteraction),cudaMemcpyHostToDevice,stream));
        CUDA_ASSERT(cudaMemcpyAsync(deviceOffsets,hostOffsets,offsets.size()*sizeof(long int),cudaMemcpyHostToDevice,stream));
    }
    cudaEvent_t done=nullptr;bool busy=false;
    void wait(){if(busy){CUDA_ASSERT(cudaEventSynchronize(done));busy=false;}}
    void clear(){wait();descriptors.clear();interactions.clear();offsets.clear();}
    void record(cudaStream_t stream){if(!done){CUDA_ASSERT(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));}CUDA_ASSERT(cudaEventRecord(done,stream));busy=true;}
    void release(){
        wait();
        if(done){CUDA_ASSERT(cudaEventDestroy(done));done=nullptr;}
        if(hostInteractions){CUDA_ASSERT(cudaFreeHost(hostInteractions));hostInteractions=nullptr;}
        if(hostOffsets){CUDA_ASSERT(cudaFreeHost(hostOffsets));hostOffsets=nullptr;}
        if(deviceInteractions){CUDA_ASSERT(cudaFree(deviceInteractions));deviceInteractions=nullptr;}
        if(deviceOffsets){CUDA_ASSERT(cudaFree(deviceOffsets));deviceOffsets=nullptr;}
        interactionsCapacity=offsetsCapacity=0;
    }
};
struct M2LGroupWorkspace {
    std::array<M2LGroupSlot,2> slots;size_t next=0;
    M2LGroupSlot& acquire(){auto& slot=slots[next++%slots.size()];slot.clear();return slot;}
    void release(){for(auto& slot:slots)slot.release();next=0;}
};
inline M2LGroupWorkspace& GetM2LGroupWorkspace(){static thread_local M2LGroupWorkspace workspace;return workspace;}

template<class KernelClass,class Cells>
__global__ void M2L_compact_group_core(typename KernelClass::CudaKernelData kernel,
    const M2LCompactArguments args,const long int count,
    const M2LInteraction* allInteractions,const long int* allOffsets){
    const long int targetIndex=blockIdx.x;
    long int member=0;
    while(member+1<count && targetIndex>=args.begin[member+1])++member;
    const auto d=args.members[member];
    const long int index=targetIndex-args.begin[member];
    if(index>=d.targets)return;
    const auto* interactions=allInteractions+d.interaction;
    const auto* offsets=allOffsets+d.offset;
    Cells targets(d.target);const auto& p=d.source;
    const Cells sources(const_cast<unsigned char*>(p[0].first),p[0].second,
        const_cast<unsigned char*>(p[1].first),p[1].second,nullptr,0);
    const long int begin=offsets[index],target=interactions[begin].target;
    KernelClass::M2LCuda(kernel,targets.getCellSymbData(target),d.level,
        M2LNeighbors<Cells>{&sources,interactions+begin},M2LPositions{interactions+begin},
        offsets[index+1]-begin,targets.getCellLocal(target));
}
template<size_t Capacity,class KernelClass,class Cells>
__global__ void M2L_wide_group_core(typename KernelClass::CudaKernelData kernel,
    const M2LWideArguments<Capacity> args,const long int count,
    const M2LInteraction* allInteractions,const long int* allOffsets){
    const long int targetIndex=blockIdx.x;long int member=0;
    while(member+1<count&&targetIndex>=args.begin[member+1])++member;
    const auto d=args.members[member];const long int index=targetIndex-args.begin[member];
    if(index>=d.targets)return;
    const auto* interactions=allInteractions+d.interaction;const auto* offsets=allOffsets+d.offset;
    Cells targets(d.target);const auto& p=d.source;
    const Cells sources(const_cast<unsigned char*>(p[0].first),p[0].second,
        const_cast<unsigned char*>(p[1].first),p[1].second,nullptr,0);
    const long int begin=offsets[index],target=interactions[begin].target;
    KernelClass::M2LCuda(kernel,targets.getCellSymbData(target),d.level,
        M2LNeighbors<Cells>{&sources,interactions+begin},M2LPositions{interactions+begin},
        offsets[index+1]-begin,targets.getCellLocal(target));
}

#endif
