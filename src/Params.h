#ifndef PARAMS_H
#define PARAMS_H

struct WindowParams {
	unsigned w;
	unsigned h;
	const char* name;
};

struct NBodySystemParams {
	unsigned bodies;
	float softening;
	float dumping;
	float deltaTime;
};

struct DeviceParams {
	unsigned blockSize;
};

#endif // !PARAMS_H