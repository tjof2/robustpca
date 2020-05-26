# C++ compiler
CXX = g++

# Compiler needs to be C++11-enabled, try -O2 or -O3
CXXFLAGS = -O3 -fPIC -Wall -std=c++11 -march=native

# Build a shared library
LDFLAGS = -shared

# Link to BLAS (or replacement), LAPACK, and Armadillo
LFLAGS = -lopenblas -llapack -larmadillo

# Build target
TARGET = librosl.so
SRCS = rosl.cpp
OBJS = $(SRCS:.cpp=.o)

.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) $^ $(LFLAGS) -o $@

.PHONY: clean
clean:
	rm *.so *.o
