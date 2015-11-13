CXX = /usr/bin/g++-4.8
CXXFLAGS = -O3 -fPIC -Wall -std=c++11 -march=native
LDFLAGS = -shared
LFLAGS = -lopenblas -llapack -larmadillo

MAJ_VERSION = 0
MIN_VERSION = 1

TARGET = libpyrosl.so.$(MAJ_VERSION).$(MIN_VERSION)
SRCS = pyrosl.cpp
OBJS = $(SRCS:.cpp=.o)

.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) $^ $(LFLAGS) -o $@
	
.PHONY: clean
clean:
	rm *~ *.so *.o
