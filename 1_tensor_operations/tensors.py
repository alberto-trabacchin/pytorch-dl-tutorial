import torch

if __name__ == "__main__":

    # Tensor types
    float_16_tensor = torch.tensor([1, 2, 3], 
                                   dtype=torch.float16,  # Data type of the tensor
                                   device='cuda',        # Device to store the tensor
                                   requires_grad=False)  # Whether to track the gradients of the tensor
    
    float_32_tensor = torch.tensor([1, 2, 3], 
                                   dtype=torch.float32,
                                   device='cuda',
                                   requires_grad=False)
    
    product_tensor = float_16_tensor * float_32_tensor
    casted_tensor = product_tensor.type(torch.int32)

    print(float_16_tensor.dtype)
    print(float_32_tensor.dtype)
    print(product_tensor.dtype)
    print(casted_tensor.dtype)

    print(f"Tensor type: {float_16_tensor.dtype}\n" \
          f"Tensor shape: {float_16_tensor.shape}\n" \
          f"Tensor device: {float_16_tensor.device}")