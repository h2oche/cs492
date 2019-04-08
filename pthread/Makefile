CC = g++
TARGET = dmm lssolver sparse

SUB_OBJS = helper.o
CLIENT_OBJS = dmm.o $(SUB_OBJS)

all: $(TARGET)
	rm *.o

lssolver: lssolver.o helper.o
	$(CC) -O2 -g -pthread -o $@ $^ -std=c++11

dmm: dmm.o helper.o
	$(CC) -O2 -g -pthread -o $@ $^ -std=c++11

sparse: sparse.o mmreader.o
	$(CC) -O2 -g -pthread -o $@ $^ -std=c++11

lssolver.o : lssolver.cpp
	$(CC) -g -pthread -c -o $@ $^ -std=c++11

dmm.o: dmm.cpp
	$(CC) -O2 -g -pthread -c -o $@ $^ -std=c++11

helper.o: helper.cpp
	$(CC) -g -c -o $@ $^ -std=c++11

mmreader.o: mmreader.cpp
	$(CC) -O2 -g -pthread -c -o $@ $^ -std=c++11

sparse.o: sparse.cpp
	$(CC) -O2 -c -g -pthread -c -o $@ $^ -std=c++11

clean:
	rm $(TARGET)
	rm *.o
	rm *.o.tmp
