CXX := /usr/bin/g++-4.8
CXXFLAGS = -O3 -fPIC -Wall -std=c++11 -march=native
LDFLAGS = -shared
LFLAGS = -lstdc++ -larmadillo

TARGET := librosl.so
SRCS = rosl.cpp
OBJS = $(SRCS:.cpp=.o)

.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) $^ $(LFLAGS) -o $@
	
.PHONY: clean
clean:
	rm *~ *.so *.o
