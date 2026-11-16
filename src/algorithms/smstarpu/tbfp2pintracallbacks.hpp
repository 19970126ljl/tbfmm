// Included inside TbfSmStarpuCallbacksCuda, preserving its existing friendship.
template<class ThisClass>
struct P2PIntraTaskArgs {
    ThisClass* algorithm;
    typename ThisClass::VecOfIndexes* indexes;
    unsigned char* data;
    size_t size;
};
template<class ThisClass>
static P2PIntraTaskArgs<ThisClass> P2PIntraUnpack(void* argument){
    P2PIntraTaskArgs<ThisClass> args{};
    starpu_codelet_unpack_args(argument,&args.algorithm,&args.indexes,&args.data,&args.size);
    return args;
}
template<class ThisClass,class Particles>
static void P2PIntraFill(TbfGroupKernelInterfaceCuda_core::P2PIntraSlot& slot,void** buffers,void* argument){
    auto args=P2PIntraUnpack<ThisClass>(argument);
    const Particles group(args.data,args.size,nullptr,0,true);
    TbfGroupKernelInterfaceCuda_core::P2PIntraPointers pointers;
    for(unsigned i=0;i<2;++i)pointers[i]={reinterpret_cast<unsigned char*>(STARPU_VARIABLE_GET_PTR(buffers[i])),STARPU_VARIABLE_GET_ELEMSIZE(buffers[i])};
    slot.append(group,*args.indexes,pointers);
}
template<class ThisClass,class Particles>
static void P2PIntraLaunch(TbfGroupKernelInterfaceCuda_core::P2PIntraSlot& slot,ThisClass* algorithm,int worker,cudaStream_t stream){
    using namespace TbfGroupKernelInterfaceCuda_core;
    using Kernel=typename ThisClass::KernelClass;
    for(int phase=0;phase<3;++phase)if(slot.arguments.total[phase]){

        P2P_intra_phase_core<Kernel,Particles><<<slot.arguments.total[phase],256,0,stream>>>(algorithm->kernels[worker].getCudaKernelData(),slot.arguments,slot.deviceItems,slot.deviceOffsets,phase);
        CUDA_ASSERT(cudaGetLastError());
    }
    slot.record(stream);
}
template<class ThisClass,class Particles>
static void P2PIntraAsyncCallback(void* buffers[],void* argument){


    auto& slot=TbfGroupKernelInterfaceCuda_core::GetP2PIntraWorkspace().acquire();
    P2PIntraFill<ThisClass,Particles>(slot,buffers,argument);
    auto stream=starpu_cuda_get_local_stream();slot.upload(stream);
    P2PIntraLaunch<ThisClass,Particles>(slot,P2PIntraUnpack<ThisClass>(argument).algorithm,starpu_worker_get_id(),stream);
}
template<class ThisClass>
static bool P2PIntraCompatible(const starpu_task* const* members,size_t count,const starpu_coarsening_target*){
    if(!count || count>4)return false;
    const auto first=P2PIntraUnpack<ThisClass>(members[0]->cl_arg);
    for(size_t m=0;m<count;++m){
        const auto* task=members[m];
        if(task->cl!=members[0]->cl || task->priority!=members[0]->priority ||
           P2PIntraUnpack<ThisClass>(task->cl_arg).algorithm!=first.algorithm)return false;
        if(STARPU_TASK_GET_NBUFFERS(task)!=2)return false;
        for(size_t n=0;n<m;++n)for(unsigned i=0;i<2;++i)for(unsigned j=0;j<2;++j)
            if(STARPU_TASK_GET_HANDLE(task,i)==STARPU_TASK_GET_HANDLE(members[n],j) &&
               ((STARPU_TASK_GET_MODE(task,i)|STARPU_TASK_GET_MODE(members[n],j))&STARPU_W))return false;
    }
    return true;
}
template<class ThisClass,class Particles>
static int P2PIntraGroupPrepare(size_t count,const starpu_coarsening_member* members,starpu_coarsening_context* context){

    auto& slot=TbfGroupKernelInterfaceCuda_core::GetP2PIntraWorkspace().acquire();context->variant_state=&slot;
    for(size_t i=0;i<count;++i)P2PIntraFill<ThisClass,Particles>(slot,members[i].buffers,members[i].cl_arg);
    slot.upload(context->stream);return 0;
}
template<class ThisClass,class Particles>
static void P2PIntraGroupLaunch(size_t count,const starpu_coarsening_member* members,starpu_coarsening_context* context){


    auto& slot=*static_cast<TbfGroupKernelInterfaceCuda_core::P2PIntraSlot*>(context->variant_state);
    P2PIntraLaunch<ThisClass,Particles>(slot,P2PIntraUnpack<ThisClass>(members[0].cl_arg).algorithm,context->target.workerid,context->stream);
}
template<class ThisClass,class Particles>
static starpu_coarsened_variant P2PIntraVariant(const TbfStarpuCudaOptions& options){
    const auto variant=[&](){
        starpu_coarsened_variant v{};v.name="tbfmm_p2p_intra_upto4";v.min_tasks=options.minTasks;v.max_tasks=4;
        v.can_append=nullptr;v.compatible=&P2PIntraCompatible<ThisClass>;v.prepare=&P2PIntraGroupPrepare<ThisClass,Particles>;v.cuda_launch=&P2PIntraGroupLaunch<ThisClass,Particles>;
        return v;
    }();
    return variant;
}
