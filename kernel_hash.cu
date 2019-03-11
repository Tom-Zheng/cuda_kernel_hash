#include <stdio.h>

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

typedef unsigned int uint32_t;

const int INITIAL_SIZE = 256;

template <typename T> 
__device__ void swap ( T& a, T& b )
{
  T c(a);
  a = b;
  b = c;
}


template <typename Key, typename Value>
class RH_hash_table  {
public:
    Key* buffer_keys;
    Value* buffer_values;
    uint32_t* buffer_hash;

    int num_elems;
    int capacity;
    uint32_t mask;

    int mutex;

    __host__  RH_hash_table()  {
        num_elems = 0;
        capacity = INITIAL_SIZE;
        mask = capacity - 1;
        mutex = 0;
        gpuErrchk(cudaMalloc((void**)&buffer_keys,  sizeof(Key) * capacity));
        gpuErrchk(cudaMalloc((void**)&buffer_values,  sizeof(Value) * capacity));
        gpuErrchk(cudaMalloc((void**)&buffer_hash,  sizeof(uint32_t) * capacity));
        gpuErrchk(cudaMemset(buffer_hash, 0, sizeof(uint32_t) * capacity));
    }

    __host__ ~RH_hash_table()  {
        gpuErrchk(cudaFree(buffer_keys));
        gpuErrchk(cudaFree(buffer_values));
        gpuErrchk(cudaFree(buffer_hash));
    }

    __device__ static uint32_t hash_key(const Key& key)  {
        // TODO
        uint32_t h = key * 2654435761;
		// MSB is used to indicate a deleted elem, so
		// clear it
		h &= 0x7fffffff;

		// Ensure that we never return 0 as a hash,
		// since we use 0 to indicate that the elem has never
		// been used at all.
		h |= h==0;
		return h; 
    }
    
	__device__ static bool is_deleted(uint32_t hash)
	{
		// MSB set indicates that this hash is a "tombstone"
		return (hash >> 31) != 0;
	}

	__device__ int desired_pos(uint32_t hash) const
	{
		return hash & mask;
	}

	__device__ int probe_distance(uint32_t hash, uint32_t slot_index) const
	{	
		return (slot_index + capacity - desired_pos(hash)) & mask;
	}

    __device__ int lookup_index(const Key& key) const
	{
		const uint32_t hash = hash_key(key);
		int pos = desired_pos(hash);
		int dist = 0;
		for(;;)
		{							
			if (buffer_hash[pos] == 0) 
				return -1;
			else if (dist > probe_distance(buffer_hash[pos], pos)) 
				return -1;
			else if (buffer_hash[pos] == hash && buffer_keys[pos] == key) 
				return pos;				

			pos = (pos+1) & mask;
			++dist;
		}
	}

    __device__ void insert(Key key, Value val) {
        num_elems++;
        if(num_elems > capacity * 0.9)  {
            printf("Hash load rate greater than 0.9\n");
            return;
        }
        uint32_t hash = hash_key(key);
		int pos = desired_pos(hash);
		int dist = 0;
		for(;;)
		{			
			if(buffer_hash[pos] == 0)  // If not occupied
			{			
                // Copy data to buffer
                buffer_keys[pos] = key;
                buffer_values[pos] = val;
                buffer_hash[pos] = hash;
				return;
			}

			// If the existing elem has probed less than us, then swap places with existing
			// elem, and keep going to find another slot for that elem.
			int existing_elem_probe_dist = probe_distance(buffer_hash[pos], pos);
			if (existing_elem_probe_dist < dist)
			{	
				if(is_deleted(buffer_hash[pos]))
				{
                    buffer_keys[pos] = key;
                    buffer_values[pos] = val;
                    buffer_hash[pos] = hash;
					return;
				}

				swap(hash, buffer_hash[pos]);
				swap(key, buffer_keys[pos]);
				swap(val, buffer_values[pos]);
				dist = existing_elem_probe_dist;				
			}

			pos = (pos+1) & mask;
			++dist;			
		}
    }

    __device__ void cuda_class_test()  {
        printf("cuda_class_test\n");
    }

    __device__ int size() {
        return num_elems;
    }

    __device__ Value* find(const Key& key)  {
		const int ix = lookup_index(key);
		return ix != -1 ? &buffer_values[ix] : NULL;
    }

	__device__ bool erase(const Key& key)
	{
		const uint32_t hash = hash_key(key);
		const int ix = lookup_index(key);

		if (ix == -1) return false;

		buffer_hash[ix] |= 0x80000000; // mark as deleted
		--num_elems;
		return true;
    }
    
    __device__ void lock()  {
        while(atomicCAS(&mutex, 0, 1) != 0) {}
    }

    __device__ void unlock()  {
        atomicExch(&mutex, 0);
    }
};




__global__ void cuda_hello(RH_hash_table<uint32_t, uint32_t> * d_hashtables, int *nBlocks){
    if(threadIdx.x == 0)  {
        d_hashtables[0].lock();
        *nBlocks = *nBlocks + 1;
        d_hashtables[0].unlock();
    }
}

int main() {
    int hash_table_num = 10;
    RH_hash_table<uint32_t, uint32_t> h_hashtables[hash_table_num];
    
    RH_hash_table<uint32_t, uint32_t> * d_hashtables = NULL;  
    gpuErrchk(cudaMalloc((void**)&d_hashtables, sizeof(RH_hash_table<uint32_t, uint32_t>) * hash_table_num));
    gpuErrchk(cudaMemcpy(d_hashtables, h_hashtables, sizeof(RH_hash_table<uint32_t, uint32_t>) * hash_table_num, cudaMemcpyHostToDevice));

    int* d_nBlocks = NULL;
    gpuErrchk(cudaMalloc((void**)&d_nBlocks, sizeof(int)));
    gpuErrchk(cudaMemset(d_nBlocks, 0, sizeof(int)));

    // Stopwatch
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // Actual work
    cuda_hello<<<10000,1>>>(d_hashtables, d_nBlocks);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf( "Time elapsed: %3.1f ms\n", elapsedTime);
    
    int nBlocks = 0; 
    gpuErrchk(cudaMemcpy(&nBlocks, d_nBlocks, sizeof(int), cudaMemcpyDeviceToHost));
    printf("nBlocks = %d\n", nBlocks);

    gpuErrchk(cudaFree( d_hashtables ));
    gpuErrchk(cudaFree( d_nBlocks ));

    return 0;
}
