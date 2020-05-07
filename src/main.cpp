#include "Handler.h"

int main(int argc, char* argv) {
	Handler::getInstance()->init(&argc, &argv, {720, 480, "nbody system"}, {30000, 0.0001f, 0.95f, 0.0001}, {256});
	Handler::getInstance()->start();
	return 0;
}