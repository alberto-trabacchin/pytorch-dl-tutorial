import torch
import time

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No GPU found")
        exit()
    tensor = torch.rand(size = (int(1e6),), 
                        dtype = torch.float32, 
                        device = 'cpu')
    
    start = time.time_ns()
    value = 0
    for i in range(len(tensor)):
        value += tensor[i] * tensor[i]
    end = time.time_ns()
    print(f"Time taken for CPU: {(end - start) / 1e6 :.2f} ms")

    cuda_tensor = tensor.to("cuda")
    start = time.time_ns()
    value = torch.matmul(cuda_tensor, cuda_tensor)
    end = time.time_ns()
    print(f"Time taken for GPU: {(end - start) / 1e6 :.2f} ms")