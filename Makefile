EXECUTABLE := ./oim

CXX := g++

CXX_SRCS := $(wildcard src/*.cpp)
OBJS := ${CXX_SRCS:.cpp=.o}

INCLUDE_DIRS := ./ /usr/local/include/
LIBRARY_DIRS :=
LIBRARIES :=

CPPFLAGS += -std=c++14 -W -Wall -O3 -march=native -mtune=native

CPPFLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir))
LDFLAGS += $(foreach library,$(LIBRARIES),-l$(library))

.PHONY: all clean

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJS)
	$(LINK.cc) $(OBJS) -o $(EXECUTABLE)

clean:
	@- $(RM) $(EXECUTABLE)
	@- $(RM) $(OBJS)
