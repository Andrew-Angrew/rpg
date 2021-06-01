CC=g++
CFLAGS=-fopenmp -O3 -march=native --std=c++17
DEPS=hnswalg.h  hnswlib.h  L2space.h  visited_list_pool.h

all: RPG

RPG: RPG.cpp $(DEPS)
	$(CC) RPG.cpp -o RPG $(CFLAGS)

clean:
	rm RPG
