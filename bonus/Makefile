MAGMADIR ?= /home/nkrajina/magma-2.0.2
CIMGDIR  ?= /home/nkrajina/CImg
CUDADIR  ?= /usr/local/cuda

MAGMA_FLAGS     := -DADD_ -I$(MAGMADIR)/include
MAGMA_LIBS              := -L${MAGMADIR}/lib -L/${CUDADIR}/lib64 -lcublas -lmagma -lcudart -lblas -lcusparse
CIMG_FLAGS              := -I${CIMGDIR}/
CIMG_LIBS               := -L/usr/X11R6/lib -lm -lpthread -lX11 -Dcimg_display=0


default:
        nvcc ef.cu -o ef $(CIMG_FLAGS) ${CIMG_LIBS} $(MAGMA_FLAGS) $(MAGMA_LIBS) -Xcompiler -Wfatal-errors -w

