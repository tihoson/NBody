#ifndef HANDLER_H
#define HANDLER_H

#include "Drawer.h"
#include "Solver.h"
#include "SolverCuda.h"
#include "Params.h"

#include <cuda_runtime_api.h>

enum class CameraMovementType {
	ROTATE,
	TRANSLATE,
	ZOOM
};

struct Camera {
	int lastX = 0, lastY = 0;
	float cameraRotation[3] = { 0, 0, 0 };
	float cameraRotationLag[3] = { 0, 0, 0 };
	float cameraTranlation[3] = { 0, 0, 0 };
	float cameraTranlationLag[3] = { 0, 0, 0 };
	float sensitivity = 0.1f;
};

class Handler {
	public:
		static Handler* getInstance();
		static bool destroy();

		void init(int* argc, char** argv, WindowParams wParams, NBodySystemParams nParams, DeviceParams dParams, const char* path = "");
		void start();
		void update();
		void draw();

		bool isPaused();
		void changePauseState();

		float getTimeStep();
		void setTimeStep(const float timeStep);

		void moveCamera(int x, int y);
		void updateCamera();
		Camera getCamera();
		void setCameraLastPositions(int x, int y);

		CameraMovementType getCameraMovmentType();
		void setCameraMovmentType(CameraMovementType type);

		void setMouseDown(bool down);

	private:
		bool paused;

		CameraMovementType cameraMovementType;
		bool mouseDown;

		float timeStep;
		Camera camera;

		WindowParams wParams;
		NBodySystemParams nParams;
		DeviceParams dParams;

		Drawer* drawer;
		Solver* solver;

	private:
		static Handler* self;
		Handler();
		~Handler();
};

// glut functions
void display();
void idle();
void motion(int x, int y);
void mouse(int button, int state, int x, int y);
void reshape(int w, int h);
void keyboard(unsigned char key, int x, int y);

#endif // !HANDLER_H

