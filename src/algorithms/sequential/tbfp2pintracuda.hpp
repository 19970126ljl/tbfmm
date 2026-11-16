#ifndef TBFP2PINTRACUDA_HPP
#define TBFP2PINTRACUDA_HPP
// Included in TbfGroupKernelInterfaceCuda_core after P2PItem.
using P2PIntraPointers=std::array<std::pair<unsigned char*,size_t>,2>;
struct P2PIntraMember {P2PIntraPointers pointers;long leaves;};
struct P2PIntraArguments {
    P2PIntraMember members[4];
    long offset[2][4]{},interaction[2][4]{},begin[3][4]{},count=0,total[3]{};
};
struct P2PIntraSlot {
    P2PIntraArguments arguments{};
    std::vector<P2PItem> items,temporary;
    std::vector<long> offsets;
    P2PItem *hostItems=nullptr,*deviceItems=nullptr;
    long *hostOffsets=nullptr,*deviceOffsets=nullptr;
    size_t itemCapacity=0,offsetCapacity=0;
    cudaEvent_t done=nullptr;
    bool busy=false;
    void wait(){
        if(busy){
            auto status=cudaEventQuery(done);
            if(status==cudaErrorNotReady){CUDA_ASSERT(cudaEventSynchronize(done));}
            else {CUDA_ASSERT(status);}
            busy=false;
        }
    }
    void acquire(){

        wait();arguments={};items.clear();offsets.clear();
    }
    template<class Particles,class Indexes>
    void append(const Particles& group,const Indexes& indexes,const P2PIntraPointers& pointers){

        if(arguments.count>=4)throw std::runtime_error("Too many intra P2P members");
        const long member=arguments.count++;
        arguments.members[member]={pointers,group.getNbLeaves()};
        temporary.clear();temporary.reserve(indexes.size());
        for(const auto& x:indexes){
            auto source=group.getElementFromSpacialIndex(x.indexSrc);
            if(!source)throw std::runtime_error("Missing intra P2P source");
            temporary.push_back({*source,x.globalTargetPos,x.arrayIndexSrc});
        }
        std::stable_sort(temporary.begin(),temporary.end(),[](const auto& a,const auto& b){return a.target<b.target;});
        for(int phase=0;phase<2;++phase){
            if(phase){
                for(auto& x:temporary)std::swap(x.source,x.target);
                std::sort(temporary.begin(),temporary.end(),[](const auto& a,const auto& b){return a.target<b.target;});
            }
            arguments.begin[phase][member]=arguments.total[phase];
            arguments.interaction[phase][member]=items.size();
            arguments.offset[phase][member]=offsets.size();
            long previous=-1;
            for(long i=0;i<long(temporary.size());++i)if(temporary[i].target!=previous){
                previous=temporary[i].target;offsets.push_back(i);++arguments.total[phase];
            }
            offsets.push_back(temporary.size());
            items.insert(items.end(),temporary.begin(),temporary.end());
        }
        arguments.begin[2][member]=arguments.total[2];
        arguments.total[2]+=group.getNbLeaves();
    }
    void upload(cudaStream_t stream){

        M2LWorkspace::Slot::reserve(hostItems,deviceItems,itemCapacity,items.size());
        M2LWorkspace::Slot::reserve(hostOffsets,deviceOffsets,offsetCapacity,offsets.size());
        std::copy(items.begin(),items.end(),hostItems);
        std::copy(offsets.begin(),offsets.end(),hostOffsets);
        if(!items.empty())CUDA_ASSERT(cudaMemcpyAsync(deviceItems,hostItems,items.size()*sizeof(P2PItem),cudaMemcpyHostToDevice,stream));
        if(!offsets.empty())CUDA_ASSERT(cudaMemcpyAsync(deviceOffsets,hostOffsets,offsets.size()*sizeof(long),cudaMemcpyHostToDevice,stream));
    }
    void record(cudaStream_t stream){
        if(!done)CUDA_ASSERT(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));
        CUDA_ASSERT(cudaEventRecord(done,stream));busy=true;
    }
    void release(){
        wait();
        if(done){CUDA_ASSERT(cudaEventDestroy(done));done=nullptr;}
        if(hostItems){CUDA_ASSERT(cudaFreeHost(hostItems));hostItems=nullptr;}
        if(deviceItems){CUDA_ASSERT(cudaFree(deviceItems));deviceItems=nullptr;}
        if(hostOffsets){CUDA_ASSERT(cudaFreeHost(hostOffsets));hostOffsets=nullptr;}
        if(deviceOffsets){CUDA_ASSERT(cudaFree(deviceOffsets));deviceOffsets=nullptr;}
        itemCapacity=offsetCapacity=0;
    }
};
struct P2PIntraWorkspace {
    std::array<P2PIntraSlot,2> slots;
    size_t next=0;
    P2PIntraSlot& acquire(){auto& slot=slots[next++%slots.size()];slot.acquire();return slot;}
    void release(){for(auto& slot:slots)slot.release();next=0;}
};
inline P2PIntraWorkspace& GetP2PIntraWorkspace(){static thread_local P2PIntraWorkspace workspace;return workspace;}
template<class Kernel,class Particles>
__global__ void P2P_intra_phase_core(typename Kernel::CudaKernelData kernel,const P2PIntraArguments arguments,
                                    const P2PItem* items,const long* offsets,int phase){
    long member=0;
    while(member+1<arguments.count&&long(blockIdx.x)>=arguments.begin[phase][member+1])++member;
    const long index=long(blockIdx.x)-arguments.begin[phase][member];
    Particles group(arguments.members[member].pointers);
    if(phase==2){
        const auto& data=TbfUtils::make_const(group).getParticleData(index);
        auto&& rhs=group.getParticleRhs(index);
        Kernel::P2PInnerCuda(kernel,group.getLeafSymbData(index),group.getParticleIndexes(index),data,rhs,group.getNbParticlesInLeaf(index));
    }else{
        const auto* os=offsets+arguments.offset[phase][member];
        const auto* is=items+arguments.interaction[phase][member];
        for(long q=os[index];q<os[index+1];++q){
            const auto x=is[q];
            const auto& source=TbfUtils::make_const(group).getParticleData(x.source);
            const auto& target=TbfUtils::make_const(group).getParticleData(x.target);
            auto&& rhs=group.getParticleRhs(x.target);
            Kernel::P2PTsmCuda(kernel,group.getLeafSymbData(x.source),group.getParticleIndexes(x.source),source,group.getNbParticlesInLeaf(x.source),
                             group.getLeafSymbData(x.target),group.getParticleIndexes(x.target),target,rhs,group.getNbParticlesInLeaf(x.target),x.direction);
        }
    }
}
#endif
