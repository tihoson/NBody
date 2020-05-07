#include "Handler.h"
#include "Header.h"
#include "Utils.h"
#include <cstring>

Handler* Handler::self = nullptr;

Handler* Handler::getInstance() {
	if (self == nullptr)
		self = new Handler;
	return self;
}

bool Handler::destroy() {
	if (self != nullptr) {
		delete self;
		self = nullptr;
		return true;
	}
	return false;
}

Handler::Handler() {
	paused = false;
	cameraMovementType = CameraMovementType::ROTATE;
	mouseDown = false;
	drawer = nullptr;
	solver = nullptr;
}

Handler::~Handler() {
	delete solver;
	delete drawer;
}

void Handler::init(int* argc, char** argv, WindowParams wParams, NBodySystemParams nParams, DeviceParams dParams, const char* path) {
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(wParams.w, wParams.h);
	glutCreateWindow(wParams.name);
	glewInit();
	glEnable(GL_DEPTH_TEST);
	glClearColor(0, 0, 0, 1);

	self->wParams = wParams;
	self->nParams = nParams;
	self->dParams = dParams;

	self->drawer = new Drawer(0.5);
	self->solver = new SolverCuda(nParams.bodies, nParams.dumping, nParams.softening, dParams.blockSize);

	float* pos = new float[4 * nParams.bodies];
	float* vel = new float[3 * nParams.bodies];

	if (!strcmp(path, ""))
		randomizeBodies(pos, vel, nParams.bodies);
	else
		readFromFile(pos, vel, nParams.bodies, path);

	self->solver->setArray(ArrayType::POSITION, pos);
	self->solver->setArray(ArrayType::VELOCITY, vel);
}

void Handler::start() {
	glutDisplayFunc(display);
	glutIdleFunc(idle);
	glutMotionFunc(motion);
	glutMouseFunc(mouse);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutMainLoop();
}

float Handler::getTimeStep() {
	return self->timeStep;
}

void Handler::setTimeStep(const float timeStep) {
	self->timeStep = timeStep;
}

void Handler::moveCamera(int x, int y) {
	float dx = (float)(x - self->camera.lastX);
	float dy = (float)(y - self->camera.lastY);

	if (self->mouseDown)
		switch (self->cameraMovementType) {
			case CameraMovementType::ROTATE:
				self->camera.cameraRotation[0] += dy / 5;
				self->camera.cameraRotation[1] += dx / 5;
				break;
			case CameraMovementType::TRANSLATE:
				self->camera.cameraTranlation[0] += dx / 100;
				self->camera.cameraTranlation[1] -= dy / 100;
				break;
			case CameraMovementType::ZOOM:
				self->camera.cameraTranlation[2] += dy / 50;
				break;
		}
}

void Handler::updateCamera() {
	for (int i = 0; i < 3; i++) {
		self->camera.cameraTranlationLag[i] += (self->camera.cameraTranlation[i] - self->camera.cameraTranlationLag[i]) * self->camera.sensitivity;
		self->camera.cameraRotationLag[i] += (self->camera.cameraRotation[i] - self->camera.cameraRotationLag[i]) * self->camera.sensitivity;
	}
}

Camera Handler::getCamera() {
	return self->camera;
}

void Handler::setCameraLastPositions(int x, int y) {
	self->camera.lastX = x;
	self->camera.lastY = y;
}

CameraMovementType Handler::getCameraMovmentType() {
	return self->cameraMovementType;
}

void Handler::setCameraMovmentType(CameraMovementType type) {
	self->cameraMovementType = type;
}

void Handler::setMouseDown(bool down) {
	self->mouseDown = down;
}

void Handler::update() {
	self->solver->update(self->nParams.deltaTime);
}

void Handler::draw() {
	self->drawer->setPositions(self->solver->getCurrentBuffer(), self->solver->getNumBodies());
	self->drawer->update();
}

bool Handler::isPaused() {
	return self->paused;
}

void Handler::changePauseState() {
	self->paused = !self->paused;
}

void display() {
	if (!Handler::getInstance()->isPaused())
		Handler::getInstance()->update();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	Handler::getInstance()->updateCamera();

	float* transLag = Handler::getInstance()->getCamera().cameraTranlationLag;
	float* rotLag = Handler::getInstance()->getCamera().cameraRotationLag;

	glTranslatef(transLag[0], transLag[1], transLag[2]);
	glRotatef(rotLag[0], 1.0, 0.0, 0.0);
	glRotatef(rotLag[1], 0.0, 1.0, 0.0);

	Handler::getInstance()->draw();

	glutSwapBuffers();
}

void idle() {
	glutPostRedisplay();
}

void motion(int x, int y) {
	Handler::getInstance()->moveCamera(x, y);

	Handler::getInstance()->setCameraLastPositions(x, y);

	glutPostRedisplay();
}

void mouse(int button, int state, int x, int y) {
	if (state == GLUT_DOWN)
		Handler::getInstance()->setMouseDown(true);
	else if (state == GLUT_UP)
		Handler::getInstance()->setMouseDown(false);

	Handler::getInstance()->setCameraLastPositions(x, y);

	glutPostRedisplay();
}

void reshape(int w, int h) {
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (float)w / (float)h, 0.1, 1000.0);
	glMatrixMode(GL_MODELVIEW);
	glViewport(0, 0, w, h);
}

void keyboard(unsigned char key, int x, int y) {
	switch (key) {
		case 'p':
		case 'P':
			Handler::getInstance()->changePauseState();
			break;
		case 'r':
		case 'R':
			Handler::getInstance()->setCameraMovmentType(CameraMovementType::ROTATE);
			break;
		case 't':
		case 'T':
			Handler::getInstance()->setCameraMovmentType(CameraMovementType::TRANSLATE);
			break;
		case 'z':
		case 'Z':
			Handler::getInstance()->setCameraMovmentType(CameraMovementType::ZOOM);
			break;
	}
}