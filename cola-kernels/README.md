********** The project is an implementation of CoLA kernels. ************

Kernel Name and corresponding Algorithm
fuse_kernel_1: A * B + C
fuse_kernel_2: Cholesky Inverse
fuse_kernel_3: Singular Value Decomposition
fuse_kernel_4: Arnoldi
fuse_kernel_5: Hutchinson

We have fours folders, 
1. Tests: sanity tests. Run as python/python3 <file name>
2. benchmark: benchmark code. Run as python/python3 <file name>
3. cola_kernels: actually cola kernels implementation
4. Playground: basic usage of CoLA kernels for each kernel. Run as python/python3 <file name>


Installation steps

1. Our work has been done on CUDA-3. So it is suggested to do tests on CUDA 3.
2. bash install.sh
3. source /scratch/cola-env/bin/activate
4. module load cuda-12.4  

If the bash install.sh fails, then its most likely the issue with folder name. Open install.sh file and rename the folder.

And then you can run any tests, benchmarks and playground code.




Once installation is done , to run the kernels follow the below steps.

1. Open any of the file in playground folder.
2. Change the value of N
3. Run as python/python3 <file name>

