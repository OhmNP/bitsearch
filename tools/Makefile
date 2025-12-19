# Makefile for bitcrack-ecdump

BINARY=bitcrack-ecdump
TARGET=$(BINDIR)/$(BINARY)

OBJS=ec_dump_main.o ec_dump_keygen.o ec_dump_io.o ec_dump_verify.o ec_dump_driver.o ec_dump_kernel.o

LIBS+=-lsecp256k1lib -lcudaUtil -lcryptoUtil -lutil -lLogger

all: $(TARGET)

$(TARGET): $(OBJS) | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS) $(LIBS)

ec_dump_kernel.o: ec_dump_kernel.cu ec_dump.cuh ec_dump_types.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -c ec_dump_kernel.cu

ec_dump_main.o: ec_dump_main.cpp ec_dump.h ec_dump_types.h
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c ec_dump_main.cpp

ec_dump_keygen.o: ec_dump_keygen.cpp ec_dump.h ec_dump_types.h
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c ec_dump_keygen.cpp

ec_dump_io.o: ec_dump_io.cpp ec_dump.h ec_dump_types.h
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c ec_dump_io.cpp

ec_dump_verify.o: ec_dump_verify.cpp ec_dump.h ec_dump_types.h
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c ec_dump_verify.cpp

ec_dump_driver.o: ec_dump_driver.cpp ec_dump.h ec_dump.cuh ec_dump_types.h
	$(CXX) $(CXXFLAGS) $(INCLUDE) -I$(CUDA_INCLUDE) -c ec_dump_driver.cpp

$(BINDIR):
	mkdir -p $(BINDIR)

clean:
	rm -f *.o
	rm -f $(TARGET)

.PHONY: all clean
