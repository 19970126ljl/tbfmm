// See LICENSE file at project root.
#ifndef FROTATIONM2LCUDA_HPP
#define FROTATIONM2LCUDA_HPP

#include <cuda_runtime.h>
#include <cassert>

/** Block-cooperative rotation M2L. All threads in a block process one target.
 * Rotation matrices are transposed at upload: adjacent coefficient threads
 * read adjacent entries while sharing the input expansion in shared memory.
 */
template <class RealType, int P>
struct FRotationM2LCuda {
    static constexpr int SizeArray = (P+1)*(P+2)/2;
    static constexpr int RotationSize = SizeArray*(2*P+1);
    static constexpr int ThreadsPerBlock = 128;

    struct Data {
        RealType* translation = nullptr;
        RealType* phase = nullptr;
        RealType* forward = nullptr;
        RealType* backward = nullptr;
    };

    __device__ static void RotateY(const RealType* inputReal, const RealType* inputImag,
                                  RealType* outputReal, RealType* outputImag,
                                  const RealType* matrix) {
        for(int idx = threadIdx.x; idx < SizeArray; idx += blockDim.x){
            int l = 0;
            while((l+1)*(l+2)/2 <= idx) ++l;
            const int first = l*(l+1)/2;
            RealType real = matrix[idx]*inputReal[first];
            RealType imag = matrix[idx]*inputImag[first];
            for(int k = 1; k <= l; ++k){
                real += matrix[(2*k-1)*SizeArray+idx]*inputReal[first+k];
                imag += matrix[(2*k)*SizeArray+idx]*inputImag[first+k];
            }
            outputReal[idx] = real;
            outputImag[idx] = imag;
        }
        __syncthreads();
    }

    template <class Neighbors, class Positions>
    __device__ static void Apply(const Data& data, const long int level,
                                const Neighbors& neighbors, const Positions& positions,
                                const long int nbNeighbors, RealType* output) {
        __shared__ RealType firstReal[SizeArray];
        __shared__ RealType firstImag[SizeArray];
        __shared__ RealType secondReal[SizeArray];
        __shared__ RealType secondImag[SizeArray];
        constexpr int CoefficientsPerThread = (SizeArray+ThreadsPerBlock-1)/ThreadsPerBlock;
        RealType sumReal[CoefficientsPerThread];
        RealType sumImag[CoefficientsPerThread];
        assert(blockDim.x == ThreadsPerBlock);
        for(int slot = 0, idx = threadIdx.x; idx < SizeArray; ++slot, idx += blockDim.x){
            sumReal[slot] = output[2*idx];
            sumImag[slot] = output[2*idx+1];
        }

        for(long int neighbor = 0; neighbor < nbNeighbors; ++neighbor){
            const long int direction = positions[neighbor];
            const RealType* phase = data.phase + direction*2*SizeArray;
            const RealType* input = reinterpret_cast<const RealType*>(neighbors[neighbor].get().data());
            for(int idx = threadIdx.x; idx < SizeArray; idx += blockDim.x){
                firstReal[idx] = input[2*idx]*phase[idx] - input[2*idx+1]*phase[SizeArray+idx];
                firstImag[idx] = input[2*idx]*phase[SizeArray+idx] + input[2*idx+1]*phase[idx];
            }
            __syncthreads();

            RotateY(firstReal, firstImag, secondReal, secondImag,
                    data.forward + direction*RotationSize);

            const RealType* translation = data.translation + (level*343+direction)*(P+1);
            for(int idx = threadIdx.x; idx < SizeArray; idx += blockDim.x){
                int l = 0;
                while((l+1)*(l+2)/2 <= idx) ++l;
                const int m = idx-l*(l+1)/2;
                const RealType sign = (m%2 ? RealType(-1) : RealType(1));
                RealType real = 0;
                RealType imag = 0;
                for(int j = m; j <= P-l; ++j){
                    const int source = j*(j+1)/2+m;
                    real += sign*translation[j+l]*secondReal[source];
                    imag -= sign*translation[j+l]*secondImag[source];
                }
                firstReal[idx] = real;
                firstImag[idx] = imag;
            }
            __syncthreads();

            RotateY(firstReal, firstImag, secondReal, secondImag,
                    data.backward + direction*RotationSize);

            // Match the CPU operator's inverse rotation and additive semantics.
            // Keep the target accumulation in thread-local registers across
            // neighbors; write global output only once, without atomics.
            for(int slot = 0, idx = threadIdx.x; idx < SizeArray; ++slot, idx += blockDim.x){
                sumReal[slot] += secondReal[idx]*phase[idx] - secondImag[idx]*phase[SizeArray+idx];
                sumImag[slot] += secondReal[idx]*phase[SizeArray+idx] + secondImag[idx]*phase[idx];
            }
            __syncthreads();
        }
        for(int slot = 0, idx = threadIdx.x; idx < SizeArray; ++slot, idx += blockDim.x){
            output[2*idx] = sumReal[slot];
            output[2*idx+1] = sumImag[slot];
        }
    }
};

#endif
