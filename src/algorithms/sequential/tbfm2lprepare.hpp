#ifndef TBFM2LPREPARE_HPP
#define TBFM2LPREPARE_HPP
// Included inside the CUDA wrapper namespace after M2LInteraction.
// No content survives for reuse: only vector capacity is retained.
struct M2LPrepareScratch {
    std::vector<long int> sourceIndexes,counts,prefix,cursor;
    std::vector<M2LInteraction> resolved;
};
inline M2LPrepareScratch& GetM2LPrepareScratch(){static thread_local M2LPrepareScratch scratch;return scratch;}
template<class Source,class Target,class Indexes>
void PrepareM2LLinear(const Source& source,const Target& target,const Indexes& indexes,
                      std::vector<M2LInteraction>& output,std::vector<long int>& offsets){

    auto& s=GetM2LPrepareScratch();const long ns=source.getNbCells(),nt=target.getNbCells();
    s.sourceIndexes.resize(ns);for(long i=0;i<ns;++i)s.sourceIndexes[i]=source.getCellSpacialIndex(i);
    const bool dense=ns && s.sourceIndexes.back()-s.sourceIndexes.front()==ns-1;
    s.counts.assign(nt,0);s.resolved.clear();s.resolved.reserve(indexes.size());
    for(const auto& i:indexes){
        long src;
        if(dense){src=i.indexSrc-s.sourceIndexes.front();if(src<0||src>=ns)continue;}
        else {auto it=std::lower_bound(s.sourceIndexes.begin(),s.sourceIndexes.end(),i.indexSrc);if(it==s.sourceIndexes.end()||*it!=i.indexSrc)continue;src=it-s.sourceIndexes.begin();}
        const long dst=i.globalTargetPos;
        assert(dst>=0 && dst<nt && target.getCellSpacialIndex(dst)==i.indexTarget);
        s.resolved.push_back({src,dst,i.arrayIndexSrc});++s.counts[dst];
    }
    offsets.clear();output.resize(s.resolved.size());if(output.empty())return;
    s.prefix.resize(nt+1);s.prefix[0]=0;
    for(long i=0;i<nt;++i){s.prefix[i+1]=s.prefix[i]+s.counts[i];if(s.counts[i])offsets.push_back(s.prefix[i]);}
    offsets.push_back(output.size());s.cursor=s.prefix;
    // Stable scatter retains the original order for interactions of a target.
    for(const auto& i:s.resolved)output[s.cursor[i.target]++]=i;
}
inline std::vector<long int>& GetM2LPreparedOffsets(){static thread_local std::vector<long int> offsets;return offsets;}
#endif
