// Included inside TbfSmStarpuCallbacksCuda; all methods have the existing
// friend access to the algorithm's kernel and index buffers.
template<class ThisClass>
struct M2LArguments {
    ThisClass* algorithm=nullptr;int level=-1;typename ThisClass::VecOfIndexes* indexes=nullptr;
    unsigned char *source=nullptr,*target=nullptr;size_t sourceSize=0,targetSize=0;
};
template<class ThisClass>
static auto M2LUnpack(const starpu_task* task,void* argument){
    M2LArguments<ThisClass> a;
    if(STARPU_TASK_GET_NBUFFERS(task)==4)
        starpu_codelet_unpack_args(argument,&a.algorithm,&a.level,&a.indexes,&a.source,&a.sourceSize,&a.target,&a.targetSize);
    else {
        starpu_codelet_unpack_args(argument,&a.algorithm,&a.level,&a.indexes,&a.source,&a.sourceSize);
        a.target=a.source;a.targetSize=a.sourceSize;
    }
    return a;
}
template<class ThisClass>
static bool M2LCompatible(const starpu_task* const* members,size_t count,
                           const starpu_coarsening_target*){

    auto reject=[](){return false;};
    if(!count||count>16)return reject();
    const auto first=M2LUnpack<ThisClass>(members[0],members[0]->cl_arg);
    struct Access {uintptr_t handle;unsigned mode;size_t member;};
    std::array<Access,64> accesses{};size_t used=0;
    for(size_t member=0;member<count;++member){
        const auto* task=members[member];
        if(task->cl!=members[0]->cl||task->priority!=members[0]->priority)return reject();
        if(member){const auto a=M2LUnpack<ThisClass>(task,task->cl_arg);
            if(a.algorithm!=first.algorithm||a.level!=first.level)return reject();}
        const auto buffers=STARPU_TASK_GET_NBUFFERS(task);
        if(buffers>4)return reject();
        for(unsigned i=0;i<buffers;++i)accesses[used++]={reinterpret_cast<uintptr_t>(STARPU_TASK_GET_HANDLE(task,i)),
            static_cast<unsigned>(STARPU_TASK_GET_MODE(task,i)),member};
    }
    std::sort(accesses.begin(),accesses.begin()+used,[](const auto& a,const auto& b){return a.handle<b.handle;});
    for(size_t begin=0;begin<used;){
        size_t end=begin+1;bool write=accesses[begin].mode&STARPU_W;bool several=false;
        while(end<used&&accesses[end].handle==accesses[begin].handle){
            write|=bool(accesses[end].mode&STARPU_W);several|=accesses[end].member!=accesses[begin].member;++end;
        }
        if(write&&several)return reject();
        begin=end;
    }
    return true;
}
template<class ThisClass,class Cells>
static int M2LGroupPrepare(size_t count,const starpu_coarsening_member* members,starpu_coarsening_context* context){

    using namespace TbfGroupKernelInterfaceCuda_core;
    auto& slot=GetM2LGroupWorkspace().acquire();context->variant_state=&slot;
    for(size_t member=0;member<count;++member){
        const auto& view=members[member];auto a=M2LUnpack<ThisClass>(view.task,view.cl_arg);auto** buffers=view.buffers;
        const Cells source(a.source,a.sourceSize,nullptr,0,nullptr,0);
        const Cells target(a.target,a.targetSize,nullptr,0,nullptr,0);
        std::vector<long int>& preparedOffsets=GetM2LPreparedOffsets();
        PrepareM2LLinear(source,target,*a.indexes,slot.temporary,preparedOffsets);
        M2LGroupDescriptor descriptor{};descriptor.level=a.level;
        descriptor.interaction=slot.interactions.size();descriptor.offset=slot.offsets.size();
        const bool between=view.nbuffers==4;
        descriptor.source[0]={reinterpret_cast<const unsigned char*>(STARPU_VARIABLE_GET_PTR(buffers[0])),STARPU_VARIABLE_GET_ELEMSIZE(buffers[0])};
        descriptor.source[1]={reinterpret_cast<const unsigned char*>(STARPU_VARIABLE_GET_PTR(buffers[1])),STARPU_VARIABLE_GET_ELEMSIZE(buffers[1])};
        descriptor.source[2]={nullptr,0};
        descriptor.target[0]={reinterpret_cast<unsigned char*>(STARPU_VARIABLE_GET_PTR(buffers[between?2:0])),STARPU_VARIABLE_GET_ELEMSIZE(buffers[between?2:0])};
        descriptor.target[1]={nullptr,0};
        descriptor.target[2]={reinterpret_cast<unsigned char*>(STARPU_VARIABLE_GET_PTR(buffers[between?3:2])),STARPU_VARIABLE_GET_ELEMSIZE(buffers[between?3:2])};
        if(!slot.temporary.empty()){
            slot.offsets.insert(slot.offsets.end(),preparedOffsets.begin(),preparedOffsets.end());
            descriptor.targets=slot.offsets.size()-descriptor.offset-1;
        }
        slot.descriptors.push_back(descriptor);
        slot.interactions.insert(slot.interactions.end(),slot.temporary.begin(),slot.temporary.end());
    }
    if(count<=4){if(!slot.interactions.empty())slot.uploadCompact(context->stream);else slot.compact.targets=0;}
    else if(count<=8)slot.uploadWide(slot.wide8,context->stream);
    else slot.uploadWide(slot.wide16,context->stream);
    return 0;
}
template<class ThisClass,class Cells>
static void M2LGroupLaunch(size_t count,const starpu_coarsening_member* members,starpu_coarsening_context* context){


    using namespace TbfGroupKernelInterfaceCuda_core;
    auto& slot=*static_cast<M2LGroupSlot*>(context->variant_state);
    auto a=M2LUnpack<ThisClass>(members[0].task,members[0].cl_arg);
    using Kernel=typename ThisClass::KernelClass;
    const auto kernel=a.algorithm->kernels[context->target.workerid].getCudaKernelData();
    if(count<=4){
        if(slot.compact.targets)M2L_compact_group_core<Kernel,Cells><<<slot.compact.targets,128,0,context->stream>>>(kernel,slot.compact,count,slot.deviceInteractions,slot.deviceOffsets);
    }else if(count<=8){
        if(slot.wide8.targets)M2L_wide_group_core<8,Kernel,Cells><<<slot.wide8.targets,128,0,context->stream>>>(kernel,slot.wide8,count,slot.deviceInteractions,slot.deviceOffsets);
    }else{
        if(slot.wide16.targets)M2L_wide_group_core<16,Kernel,Cells><<<slot.wide16.targets,128,0,context->stream>>>(kernel,slot.wide16,count,slot.deviceInteractions,slot.deviceOffsets);
    }
    CUDA_ASSERT(cudaGetLastError());
    // The per-execution interaction/offset data still need lifetime protection.
    slot.record(context->stream);
}
static void M2LGroupRelease(starpu_coarsening_context*){}
template<class ThisClass,class Cells,bool Between>
static starpu_coarsened_variant M2LVariant(const TbfStarpuCudaOptions& options){
    const auto variant=[&](){
        starpu_coarsened_variant v{};v.name=Between?"tbfmm_cross_m2l_upto16":"tbfmm_inside_m2l_upto4";v.min_tasks=options.minTasks;v.max_tasks=Between?options.m2lMaxTasks:4;
        v.can_append=nullptr;v.compatible=&M2LCompatible<ThisClass>;v.prepare=&M2LGroupPrepare<ThisClass,Cells>;
        v.cuda_launch=&M2LGroupLaunch<ThisClass,Cells>;v.release=&M2LGroupRelease;return v;
    }();
    return variant;
}
