#ifndef SOLVER_CUDA_H
#define SOLVER_CUDA_H

#include "Solver.h"
#include <cstdio>
#include <cstdlib>

struct DeviceData {
	float* devVel;
	float* devPos[2];
};

void updateBodies(cudaGraphicsResource** graphRes, unsigned currentBuffer,
	float deltaTime, float damping, float softening, int boides,
	DeviceData deviceData, unsigned blockSize);

void __cudaCheckError(cudaError_t err, const char* file, const int line);

#define cudaCheckError(err) __cudaCheckError(err, __FILE__, __LINE__)

class SolverCuda : public Solver {
	public:
		SolverCuda(unsigned bodies, float dumping, float softening, unsigned blockSize);
		virtual ~SolverCuda();

		virtual void update(float deltaTime) override;

		virtual void setArray(ArrayType type, const float* data) override;

		virtual unsigned getCurrentBuffer() const override {
			return pbo[currentBuffer];
		}

	private:
		void init();

	private:
		unsigned pbo[2];
		cudaGraphicsResource* graphRes[2];

		DeviceData deviceData;

		unsigned currentBuffer;

		unsigned blockSize;

		float softening;
};

#endif // !SOLVER_CUDA_H