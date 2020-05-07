#include "SolverCuda.h"

#include <cstring>

void __cudaCheckError(cudaError_t err, const char* file, const int line) {
	if (err != cudaSuccess) {
		fprintf(stderr, "Error! %s(%s) in %s at %d.\n", 
			cudaGetErrorString(err), cudaGetErrorName(err),
			file, line);
		exit(0);
	}
}

SolverCuda::SolverCuda(unsigned bodies, float dumping, float softening, unsigned blockSize):
	Solver(bodies, dumping) {

	currentBuffer = 0;
	this->softening = softening;
	this->blockSize = blockSize;

	init();
}

void SolverCuda::init() {
	const size_t size = 4 * bodies * sizeof(float);

	deviceData = DeviceData();

	float* emptyData = new float[4 * bodies];
	memset(emptyData, 0, size);

	glGenBuffers(2, pbo);

	for (int i = 0; i < 2; i++) {
		glBindBuffer(GL_ARRAY_BUFFER, pbo[i]);
		glBufferData(GL_ARRAY_BUFFER, size, emptyData, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		cudaCheckError(cudaGraphicsGLRegisterBuffer(&graphRes[i], pbo[i], cudaGraphicsMapFlagsNone));
	}

	cudaCheckError(cudaMalloc((void**)&deviceData.devVel, 3 * bodies * sizeof(float)));

	delete[] emptyData;
}

SolverCuda::~SolverCuda() {
	for (int i = 0; i < 2; i++)
		cudaCheckError(cudaGraphicsUnregisterResource(graphRes[i]));

	glDeleteBuffers(2, (const GLuint*)pbo);

	cudaCheckError(cudaFree((void*)deviceData.devVel));

}

void SolverCuda::update(float deltaTime) {
	cudaCheckError(cudaGetLastError());

	updateBodies(graphRes, currentBuffer, deltaTime, damping, softening, bodies, deviceData, blockSize);
	currentBuffer = 1 - currentBuffer;
}

void SolverCuda::setArray(ArrayType type, const float* data) {
	currentBuffer = 0;

	switch (type) {
		case ArrayType::POSITION:
			glBindBuffer(GL_ARRAY_BUFFER, pbo[currentBuffer]);
			glBufferSubData(GL_ARRAY_BUFFER, 0, 4 * sizeof(float) * bodies, data);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			break;
		case ArrayType::VELOCITY:
			cudaCheckError(cudaMemcpy((void*)deviceData.devVel, data, 3 * sizeof(float) * bodies, cudaMemcpyHostToDevice));
			break;
	}
}