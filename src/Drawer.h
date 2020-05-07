#ifndef DRAWER_H
#define DRAWER_H

#include "Header.h"

class Drawer {
	public:
		Drawer(float pointSize);
		~Drawer();

		void update();

		void setPositions(unsigned pbo, unsigned bodies);

	private:
		void init();

	private:
		unsigned bodies;

		unsigned pbo;
		unsigned pointsProgram;

		float pointSize;
};

#endif // !DRAWER_H