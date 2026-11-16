// Included inside the existing CUDA callback class, preserving friend access.
template<class ThisClass>
struct P2PTaskArgs {ThisClass* algorithm;typename ThisClass::VecOfIndexes* indexes;unsigned char *source,*target;size_t sourceSize,targetSize;};
template<class ThisClass>
static P2PTaskArgs<ThisClass> P2PUnpack(void* arg){
    P2PTaskArgs<ThisClass> a{};starpu_codelet_unpack_args(arg,&a.algorithm,&a.indexes,&a.source,&a.sourceSize,&a.target,&a.targetSize);return a;
}
template<class ThisClass,class Particles>
static void P2PFill(TbfGroupKernelInterfaceCuda_core::P2PSlot& slot,void** buffers,void* arg){
    auto a=P2PUnpack<ThisClass>(arg);const Particles source(a.source,a.sourceSize,nullptr,0,true),target(a.target,a.targetSize,nullptr,0,true);
    std::array<std::pair<unsigned char*,size_t>,2> sp,tp;
    for(unsigned i=0;i<2;++i){sp[i]={reinterpret_cast<unsigned char*>(STARPU_VARIABLE_GET_PTR(buffers[i])),STARPU_VARIABLE_GET_ELEMSIZE(buffers[i])};tp[i]={reinterpret_cast<unsigned char*>(STARPU_VARIABLE_GET_PTR(buffers[i+2])),STARPU_VARIABLE_GET_ELEMSIZE(buffers[i+2])};}
    slot.append(source,target,*a.indexes,sp,tp);
}
template<class ThisClass,class Particles>
static void P2PAsyncCallback(void* buffers[],void* arg){


    using namespace TbfGroupKernelInterfaceCuda_core;
    auto& slot=GetP2PWorkspace().acquire();P2PFill<ThisClass,Particles>(slot,buffers,arg);auto stream=starpu_cuda_get_local_stream();slot.upload(stream);
    auto a=P2PUnpack<ThisClass>(arg);
    if(slot.arguments.targets){using K=typename ThisClass::KernelClass;P2P_group_core<K,Particles><<<slot.arguments.targets,256,0,stream>>>(a.algorithm->kernels[starpu_worker_get_id()].getCudaKernelData(),slot.arguments,slot.deviceItems,slot.deviceOffsets);CUDA_ASSERT(cudaGetLastError());}
    slot.record(stream);
}
template<class ThisClass,class Particles>
static void P2PWideFill(TbfGroupKernelInterfaceCuda_core::P2PWideSlot& slot,void** buffers,void* arg){
    auto a=P2PUnpack<ThisClass>(arg);const Particles source(a.source,a.sourceSize,nullptr,0,true),target(a.target,a.targetSize,nullptr,0,true);
    std::array<std::pair<unsigned char*,size_t>,2> sp,tp;
    for(unsigned i=0;i<2;++i){sp[i]={reinterpret_cast<unsigned char*>(STARPU_VARIABLE_GET_PTR(buffers[i])),STARPU_VARIABLE_GET_ELEMSIZE(buffers[i])};tp[i]={reinterpret_cast<unsigned char*>(STARPU_VARIABLE_GET_PTR(buffers[i+2])),STARPU_VARIABLE_GET_ELEMSIZE(buffers[i+2])};}
    slot.append(source,target,*a.indexes,sp,tp);
}
template<class ThisClass>
static bool P2PCompatible(const starpu_task* const* members,size_t count,const starpu_coarsening_target*){
    if(!count||count>8)return false;
    const auto first=P2PUnpack<ThisClass>(members[0]->cl_arg);
    struct Access {uintptr_t handle;unsigned mode;size_t member;};
    std::array<Access,32> accesses{};size_t used=0;
    for(size_t m=0;m<count;++m){const auto* task=members[m];
        if(task->cl!=members[0]->cl||task->priority!=members[0]->priority)return false;
        if(m&&P2PUnpack<ThisClass>(task->cl_arg).algorithm!=first.algorithm)return false;
        const auto buffers=STARPU_TASK_GET_NBUFFERS(task);if(buffers>4)return false;
        for(unsigned i=0;i<buffers;++i)accesses[used++]={reinterpret_cast<uintptr_t>(STARPU_TASK_GET_HANDLE(task,i)),static_cast<unsigned>(STARPU_TASK_GET_MODE(task,i)),m};
    }
    std::sort(accesses.begin(),accesses.begin()+used,[](const auto& a,const auto& b){return a.handle<b.handle;});
    for(size_t begin=0;begin<used;){size_t end=begin+1;bool write=accesses[begin].mode&STARPU_W,several=false;
        while(end<used&&accesses[end].handle==accesses[begin].handle){write|=bool(accesses[end].mode&STARPU_W);several|=accesses[end].member!=accesses[begin].member;++end;}
        if(write&&several)return false;begin=end;
    }
    return true;
}
template<class ThisClass,class Particles>
static int P2PGroupPrepare(size_t count,const starpu_coarsening_member* members,starpu_coarsening_context* ctx){

    if(count>4){auto& slot=TbfGroupKernelInterfaceCuda_core::GetP2PWideWorkspace().acquire();ctx->variant_state=&slot;
        for(size_t i=0;i<count;++i)P2PWideFill<ThisClass,Particles>(slot,members[i].buffers,members[i].cl_arg);
        slot.upload(ctx->stream);return 0;
    }
    auto& slot=TbfGroupKernelInterfaceCuda_core::GetP2PWorkspace().acquire();ctx->variant_state=&slot;
    for(size_t i=0;i<count;++i)P2PFill<ThisClass,Particles>(slot,members[i].buffers,members[i].cl_arg);
    slot.upload(ctx->stream);return 0;
}
template<class ThisClass,class Particles>
static void P2PGroupLaunch(size_t count,const starpu_coarsening_member* members,starpu_coarsening_context* ctx){


    if(count>4){using namespace TbfGroupKernelInterfaceCuda_core;auto& slot=*static_cast<P2PWideSlot*>(ctx->variant_state);
        auto a=P2PUnpack<ThisClass>(members[0].cl_arg);using K=typename ThisClass::KernelClass;
        if(slot.arguments.targets){P2P_wide_group_core<K,Particles><<<slot.arguments.targets,256,0,ctx->stream>>>(a.algorithm->kernels[ctx->target.workerid].getCudaKernelData(),slot.arguments,slot.deviceItems,slot.deviceOffsets);CUDA_ASSERT(cudaGetLastError());}
        slot.record(ctx->stream);return;
    }
    using namespace TbfGroupKernelInterfaceCuda_core;auto& slot=*static_cast<P2PSlot*>(ctx->variant_state);auto a=P2PUnpack<ThisClass>(members[0].cl_arg);
    if(slot.arguments.targets){using K=typename ThisClass::KernelClass;P2P_group_core<K,Particles><<<slot.arguments.targets,256,0,ctx->stream>>>(a.algorithm->kernels[ctx->target.workerid].getCudaKernelData(),slot.arguments,slot.deviceItems,slot.deviceOffsets);CUDA_ASSERT(cudaGetLastError());}
    slot.record(ctx->stream);
}
template<class ThisClass,class Particles>
static starpu_coarsened_variant P2PVariant(const TbfStarpuCudaOptions& options){const auto v=[&](){starpu_coarsened_variant v{};v.name="tbfmm_p2p_between_upto8";v.min_tasks=options.minTasks;v.max_tasks=options.p2pMaxTasks;v.can_append=nullptr;v.compatible=&P2PCompatible<ThisClass>;v.prepare=&P2PGroupPrepare<ThisClass,Particles>;v.cuda_launch=&P2PGroupLaunch<ThisClass,Particles>;return v;}();return v;}
