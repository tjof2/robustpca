# Current fix because it won't work with GCC 5.2
#CXX = g++
CXX = /usr/bin/g++-4.8

# Compiler needs to be C++11-enabled
CXXFLAGS = -O3 -fPIC -Wall -std=c++11 -march=native

# Build a shared library
LDFLAGS = -shared

# Link to BLAS (or replacement)
LFLAGS = -lopenblas -llapack -larmadillo

#####################################
# No need to modify below this line #
#####################################

# Versioning
MAJ_VERSION = 0
MIN_VERSION = 2

# Build target
TARGET = librosl.so.$(MAJ_VERSION).$(MIN_VERSION)
SRCS = rosl.cpp
OBJS = $(SRCS:.cpp=.o)

.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) $^ $(LFLAGS) -o $@
	
.PHONY: clean
clean:
	rm *~ *.so* *.o
