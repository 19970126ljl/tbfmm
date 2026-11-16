#ifndef TBFP2PGROUPCUDA_HPP
#define TBFP2PGROUPCUDA_HPP
// Per-execution metadata, never cached across calls.
struct P2PItem {long source,target,direction;};
struct P2PDirection {
    std::array<std::pair<unsigned char*,size_t>,2> source,target;
    long interaction,offset,targets,begin;
};
struct P2PArguments {P2PDirection directions[8];long count=0,targets=0;};
struct P2PSlot {
    P2PItem *hostItems=nullptr,*deviceItems=nullptr;
    long *hostOffsets=nullptr,*deviceOffsets=nullptr;
    size_t itemCapacity=0,offsetCapacity=0;
    std::vector<P2PItem> items,resolved,ordered;
    std::vector<long> offsets,sourceIndexes,counts,prefix,cursor;
    P2PArguments arguments{};
    cudaEvent_t done=nullptr;bool busy=false;
    void acquire(){
        if(busy){CUDA_ASSERT(cudaEventSynchronize(done));busy=false;}
        items.clear();offsets.clear();arguments={};
    }
    template<class Source,class Target,class Indexes>
    void append(const Source& source,const Target& target,const Indexes& indexes,
                const std::array<std::pair<unsigned char*,size_t>,2>& sourcePtrs,
                const std::array<std::pair<unsigned char*,size_t>,2>& targetPtrs){

        const long ns=source.getNbLeaves();sourceIndexes.resize(ns);
        for(long i=0;i<ns;++i)sourceIndexes[i]=source.getLeafSpacialIndex(i);
        const bool dense=ns && sourceIndexes.back()-sourceIndexes.front()==ns-1;
        resolved.clear();resolved.reserve(indexes.size());
        for(const auto& x:indexes){
            long src;
            if(dense){src=x.indexSrc-sourceIndexes.front();if(src<0||src>=ns)continue;}
            else {auto it=std::lower_bound(sourceIndexes.begin(),sourceIndexes.end(),x.indexSrc);if(it==sourceIndexes.end()||*it!=x.indexSrc)continue;src=it-sourceIndexes.begin();}
            const long dst=x.globalTargetPos;
            assert(dst>=0 && dst<target.getNbLeaves() && target.getLeafSpacialIndex(dst)==x.indexTarget);
            resolved.push_back({src,dst,x.arrayIndexSrc});
        }
        appendDirection(target.getNbLeaves(),sourcePtrs,targetPtrs,false);
        appendDirection(source.getNbLeaves(),targetPtrs,sourcePtrs,true);
    }
    void appendDirection(long leaves,const std::array<std::pair<unsigned char*,size_t>,2>& sourcePtrs,
                         const std::array<std::pair<unsigned char*,size_t>,2>& targetPtrs,bool reverse){
        P2PDirection d{};d.source=sourcePtrs;d.target=targetPtrs;d.interaction=items.size();d.offset=offsets.size();d.begin=arguments.targets;
        counts.assign(leaves,0);
        for(const auto& i:resolved)++counts[reverse?i.source:i.target];
        prefix.resize(leaves+1);prefix[0]=0;
        for(long i=0;i<leaves;++i){prefix[i+1]=prefix[i]+counts[i];if(counts[i]){offsets.push_back(prefix[i]);++d.targets;}}
        if(!resolved.empty())offsets.push_back(resolved.size());
        cursor=prefix;ordered.resize(resolved.size());
        for(const auto& i:resolved){P2PItem x=reverse?P2PItem{i.target,i.source,i.direction}:i;ordered[cursor[x.target]++]=x;}
        items.insert(items.end(),ordered.begin(),ordered.end());
        arguments.directions[arguments.count++]=d;arguments.targets+=d.targets;
    }
    void upload(cudaStream_t stream){
        if(items.empty())return;
        M2LWorkspace::Slot::reserve(hostItems,deviceItems,itemCapacity,items.size());
        M2LWorkspace::Slot::reserve(hostOffsets,deviceOffsets,offsetCapacity,offsets.size());
        std::copy(items.begin(),items.end(),hostItems);std::copy(offsets.begin(),offsets.end(),hostOffsets);
        CUDA_ASSERT(cudaMemcpyAsync(deviceItems,hostItems,items.size()*sizeof(P2PItem),cudaMemcpyHostToDevice,stream));
        CUDA_ASSERT(cudaMemcpyAsync(deviceOffsets,hostOffsets,offsets.size()*sizeof(long),cudaMemcpyHostToDevice,stream));
    }
    void record(cudaStream_t stream){if(!done)CUDA_ASSERT(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));CUDA_ASSERT(cudaEventRecord(done,stream));busy=true;}
    void release(){
        if(busy)CUDA_ASSERT(cudaEventSynchronize(done));busy=false;
        if(done){CUDA_ASSERT(cudaEventDestroy(done));done=nullptr;}
        if(hostItems){CUDA_ASSERT(cudaFreeHost(hostItems));hostItems=nullptr;}if(deviceItems){CUDA_ASSERT(cudaFree(deviceItems));deviceItems=nullptr;}
        if(hostOffsets){CUDA_ASSERT(cudaFreeHost(hostOffsets));hostOffsets=nullptr;}if(deviceOffsets){CUDA_ASSERT(cudaFree(deviceOffsets));deviceOffsets=nullptr;}
        itemCapacity=offsetCapacity=0;
    }
};
struct P2PWorkspace {std::array<P2PSlot,2> slots;size_t next=0;P2PSlot& acquire(){auto& s=slots[next++%2];s.acquire();return s;}void release(){for(auto& s:slots)s.release();next=0;}};
inline P2PWorkspace& GetP2PWorkspace(){static thread_local P2PWorkspace workspace;return workspace;}
template<class Kernel,class Particles>
__global__ void P2P_group_core(typename Kernel::CudaKernelData kernel,const P2PArguments args,
                              const P2PItem* items,const long* offsets){
    const long work=blockIdx.x;long di=0;
    while(di+1<args.count && work>=args.directions[di+1].begin)++di;
    const auto d=args.directions[di];const long t=work-d.begin;if(t>=d.targets)return;
    Particles source(d.source),target(d.target);
    const auto* is=items+d.interaction;const auto* os=offsets+d.offset;
    for(long n=os[t];n<os[t+1];++n){const auto i=is[n];
        const auto& sd=TbfUtils::make_const(source).getParticleData(i.source);
        const auto& td=TbfUtils::make_const(target).getParticleData(i.target);auto&& rhs=target.getParticleRhs(i.target);
        Kernel::P2PTsmCuda(kernel,source.getLeafSymbData(i.source),source.getParticleIndexes(i.source),sd,source.getNbParticlesInLeaf(i.source),
                          target.getLeafSymbData(i.target),target.getParticleIndexes(i.target),td,rhs,target.getNbParticlesInLeaf(i.target),i.direction);
    }
}
// Independent wide-group ABI; original single and <=4-group path above unchanged.
struct P2PWideArguments {P2PDirection directions[16];long count=0,targets=0;};
struct P2PWideSlot {
    P2PItem *hostItems=nullptr,*deviceItems=nullptr;
    long *hostOffsets=nullptr,*deviceOffsets=nullptr;
    size_t itemCapacity=0,offsetCapacity=0;
    std::vector<P2PItem> items,resolved,ordered;
    std::vector<long> offsets,sourceIndexes,counts,prefix,cursor;
    P2PWideArguments arguments{};
    cudaEvent_t done=nullptr;bool busy=false;
    void acquire(){
        if(busy){CUDA_ASSERT(cudaEventSynchronize(done));busy=false;}
        items.clear();offsets.clear();arguments={};
    }
    template<class Source,class Target,class Indexes>
    void append(const Source& source,const Target& target,const Indexes& indexes,
                const std::array<std::pair<unsigned char*,size_t>,2>& sourcePtrs,
                const std::array<std::pair<unsigned char*,size_t>,2>& targetPtrs){

        const long ns=source.getNbLeaves();sourceIndexes.resize(ns);
        for(long i=0;i<ns;++i)sourceIndexes[i]=source.getLeafSpacialIndex(i);
        const bool dense=ns && sourceIndexes.back()-sourceIndexes.front()==ns-1;
        resolved.clear();resolved.reserve(indexes.size());
        for(const auto& x:indexes){
            long src;
            if(dense){src=x.indexSrc-sourceIndexes.front();if(src<0||src>=ns)continue;}
            else {auto it=std::lower_bound(sourceIndexes.begin(),sourceIndexes.end(),x.indexSrc);if(it==sourceIndexes.end()||*it!=x.indexSrc)continue;src=it-sourceIndexes.begin();}
            const long dst=x.globalTargetPos;
            assert(dst>=0 && dst<target.getNbLeaves() && target.getLeafSpacialIndex(dst)==x.indexTarget);
            resolved.push_back({src,dst,x.arrayIndexSrc});
        }
        appendDirection(target.getNbLeaves(),sourcePtrs,targetPtrs,false);
        appendDirection(source.getNbLeaves(),targetPtrs,sourcePtrs,true);
    }
    void appendDirection(long leaves,const std::array<std::pair<unsigned char*,size_t>,2>& sourcePtrs,
                         const std::array<std::pair<unsigned char*,size_t>,2>& targetPtrs,bool reverse){
        P2PDirection d{};d.source=sourcePtrs;d.target=targetPtrs;d.interaction=items.size();d.offset=offsets.size();d.begin=arguments.targets;
        counts.assign(leaves,0);
        for(const auto& i:resolved)++counts[reverse?i.source:i.target];
        prefix.resize(leaves+1);prefix[0]=0;
        for(long i=0;i<leaves;++i){prefix[i+1]=prefix[i]+counts[i];if(counts[i]){offsets.push_back(prefix[i]);++d.targets;}}
        if(!resolved.empty())offsets.push_back(resolved.size());
        cursor=prefix;ordered.resize(resolved.size());
        for(const auto& i:resolved){P2PItem x=reverse?P2PItem{i.target,i.source,i.direction}:i;ordered[cursor[x.target]++]=x;}
        items.insert(items.end(),ordered.begin(),ordered.end());
        arguments.directions[arguments.count++]=d;arguments.targets+=d.targets;
    }
    void upload(cudaStream_t stream){
        if(items.empty())return;
        M2LWorkspace::Slot::reserve(hostItems,deviceItems,itemCapacity,items.size());
        M2LWorkspace::Slot::reserve(hostOffsets,deviceOffsets,offsetCapacity,offsets.size());
        std::copy(items.begin(),items.end(),hostItems);std::copy(offsets.begin(),offsets.end(),hostOffsets);
        CUDA_ASSERT(cudaMemcpyAsync(deviceItems,hostItems,items.size()*sizeof(P2PItem),cudaMemcpyHostToDevice,stream));
        CUDA_ASSERT(cudaMemcpyAsync(deviceOffsets,hostOffsets,offsets.size()*sizeof(long),cudaMemcpyHostToDevice,stream));
    }
    void record(cudaStream_t stream){if(!done)CUDA_ASSERT(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));CUDA_ASSERT(cudaEventRecord(done,stream));busy=true;}
    void release(){
        if(busy)CUDA_ASSERT(cudaEventSynchronize(done));busy=false;
        if(done){CUDA_ASSERT(cudaEventDestroy(done));done=nullptr;}
        if(hostItems){CUDA_ASSERT(cudaFreeHost(hostItems));hostItems=nullptr;}if(deviceItems){CUDA_ASSERT(cudaFree(deviceItems));deviceItems=nullptr;}
        if(hostOffsets){CUDA_ASSERT(cudaFreeHost(hostOffsets));hostOffsets=nullptr;}if(deviceOffsets){CUDA_ASSERT(cudaFree(deviceOffsets));deviceOffsets=nullptr;}
        itemCapacity=offsetCapacity=0;
    }
};
struct P2PWideWorkspace {std::array<P2PWideSlot,2> slots;size_t next=0;P2PWideSlot& acquire(){auto& s=slots[next++%2];s.acquire();return s;}void release(){for(auto& s:slots)s.release();next=0;}};
inline P2PWideWorkspace& GetP2PWideWorkspace(){static thread_local P2PWideWorkspace workspace;return workspace;}
template<class Kernel,class Particles>
__global__ void P2P_wide_group_core(typename Kernel::CudaKernelData kernel,const P2PWideArguments args,
                              const P2PItem* items,const long* offsets){
    const long work=blockIdx.x;long di=0;
    while(di+1<args.count && work>=args.directions[di+1].begin)++di;
    const auto d=args.directions[di];const long t=work-d.begin;if(t>=d.targets)return;
    Particles source(d.source),target(d.target);
    const auto* is=items+d.interaction;const auto* os=offsets+d.offset;
    for(long n=os[t];n<os[t+1];++n){const auto i=is[n];
        const auto& sd=TbfUtils::make_const(source).getParticleData(i.source);
        const auto& td=TbfUtils::make_const(target).getParticleData(i.target);auto&& rhs=target.getParticleRhs(i.target);
        Kernel::P2PTsmCuda(kernel,source.getLeafSymbData(i.source),source.getParticleIndexes(i.source),sd,source.getNbParticlesInLeaf(i.source),
                          target.getLeafSymbData(i.target),target.getParticleIndexes(i.target),td,rhs,target.getNbParticlesInLeaf(i.target),i.direction);
    }
}
#endif
