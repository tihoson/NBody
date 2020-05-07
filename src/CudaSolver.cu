#include "SolverCuda.h"

#ifndef __CUDACC__  
#define __CUDACC__
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

using cooperative_groups::sync;
using cooperative_groups::thread_block;
using cooperative_groups::this_thread_block;

__device__ void bodiesInteraction(float4 first, float4 second, float3* acc, float softening) {
	float3 r;

	r.x = second.x - first.x;
	r.y = second.y - first.y;
	r.z = second.z - first.z;

	float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + softening;
	float invDist = rsqrtf(distSqr);
	float invDistCube = invDist * invDist * invDist;
	float p = second.w * invDistCube;

	(*acc).x += r.x * p;
	(*acc).y += r.y * p;
	(*acc).z += r.z * p;
}

__device__ float3 getAcc(float4* positions, float4 pos, float softening, thread_block tb) {
	extern __shared__ float4 smem[];

	float3 acc = { 0, 0, 0 };

	for (int t = 0; t < gridDim.x; t++) {
		smem[threadIdx.x] = positions[t * blockDim.x + threadIdx.x];

		sync(tb);

		for (unsigned i = 0; i < blockDim.x; i++)
			bodiesInteraction(pos, smem[i], &acc, softening);

		sync(tb);
	}

	return acc;
}

__global__ void integrateBodies(float4* __restrict__ oldPositions, float4* __restrict__ newPositions, float3* __restrict__ velocity,
	float deltaTime, float damping, float softening, int bodies) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= bodies)
		return;

	thread_block tb = this_thread_block();

	float4 pos = oldPositions[idx];
	float3 acc = getAcc(oldPositions, pos, softening, tb);
	float3 vel = velocity[idx];

	vel.x = (vel.x + acc.x * deltaTime) * damping;
	vel.y = (vel.y + acc.y * deltaTime) * damping;
	vel.z = (vel.z + acc.z * deltaTime) * damping;

	pos.x += vel.x * deltaTime;
	pos.y += vel.y * deltaTime;
	pos.z += vel.z * deltaTime;

	velocity[idx] = vel;
	newPositions[idx] = pos;
}

void updateBodies(cudaGraphicsResource** graphRes, unsigned currentBuffer,
	float deltaTime, float damping, float softening, int bodies,
	DeviceData deviceData, unsigned blockSize) {
	cudaCheckError(cudaGraphicsMapResources(2, graphRes));
	size_t size;
	cudaCheckError(cudaGraphicsResourceGetMappedPointer((void**)&(deviceData.devPos[currentBuffer]), &size, graphRes[currentBuffer]));
	cudaCheckError(cudaGraphicsResourceGetMappedPointer((void**)&(deviceData.devPos[1 - currentBuffer]), &size, graphRes[1 - currentBuffer]));

	unsigned blocks = (bodies + blockSize - 1) / blockSize;
	size_t shared = blockSize * 4 * sizeof(float);

	integrateBodies <<< blocks, blockSize, shared >>> (
		(float4*)(deviceData.devPos[currentBuffer]),
		(float4*)(deviceData.devPos[1 - currentBuffer]),
		(float3*)(deviceData.devVel),
		deltaTime, damping, softening, bodies);

	cudaCheckError(cudaGraphicsUnmapResources(2, graphRes));
}