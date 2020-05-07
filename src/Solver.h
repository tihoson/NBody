#ifndef SOLVER_H
#define SOLVER_H

#include "Header.h"

enum class ArrayType {
    POSITION,
    VELOCITY,
};

class Solver {
    public:
        Solver(unsigned bodies, float dumping) {
            this->bodies = bodies;
            this->damping = dumping;
        }
        virtual ~Solver() {}

        virtual void update(float deltaTime) = 0;

        virtual void setArray(ArrayType type, const float* data) = 0;

        virtual unsigned getCurrentBuffer() const = 0;
        virtual unsigned getNumBodies() const {
            return bodies;
        };

    protected:
        unsigned bodies;
        float damping;
};

#endif // !SOLVER_H