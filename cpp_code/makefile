CC=g++
CFLAGS=-c -g -Wall -O2
LDFLAGS=
SOURCES=train_pmf.cpp pmf_recommender.cpp ran_sampler.cpp lin_alg.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=train_pmf

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@
	
clean:
	rm -f *.o test
