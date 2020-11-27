horovodrun \
  --verbose \
  -p 9969 \
  -np 16 -H trdma2011-h0:8,trdma2011-h1:8 \
  python run_test_rdma_mnist.py

mpirun \
    -port 9969 \
    -np 16 \
    -H trdma2011-h0:8,trdma2011-h1:8 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=DEBUG -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python run_test_rdma_mnist.py