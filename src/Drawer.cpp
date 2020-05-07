#include "Drawer.h"

Drawer::Drawer(float pointsSize) {
	this->pointSize = pointSize;

	bodies = 0;
	pbo = 0;
	pointsProgram = 0;

	init();
}

Drawer::~Drawer() {
	glDeleteProgram(pointsProgram);
}

void Drawer::update() {
	glColor3f(1, 1, 1);
	glPointSize(pointSize);
	glUseProgram(pointsProgram);
	glEnableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, pbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);
	glDrawArrays(GL_POINTS, 0, bodies);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDisableClientState(GL_VERTEX_ARRAY);
	glUseProgram(0);
}

void Drawer::setPositions(unsigned pbo, unsigned bodies) {
	this->pbo = pbo;
	this->bodies = bodies;
}

const char vertexShader[] = {
	"void main()\n"
	"{\n"
	"	vec4 vert = vec4(gl_Vertex.xyz, 1.0);\n"
	"	gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vert;\n"
	"	gl_FrontColor = gl_Color;\n"
	"}\n"
};

void Drawer::init() {
	const unsigned shader = glCreateShader(GL_VERTEX_SHADER);
	const char* vs = vertexShader;

	glShaderSource(shader, 1, &vs, 0);
	glCompileShader(shader);

	pointsProgram = glCreateProgram();
	glAttachShader(pointsProgram, shader);
	glLinkProgram(pointsProgram);

	glDeleteShader(shader);
}