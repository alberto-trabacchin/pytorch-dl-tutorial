import torch

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_batch = torch.rand(32, 3, 64, 64)
    conv_layer = torch.nn.Conv2d(in_channels = 3,
                                 out_channels = 10,
                                 kernel_size = 5,
                                 stride = 1,
                                 padding = 1).to(device)
    max_pool_layer = torch.nn.MaxPool2d(kernel_size = 2,
                                        stride = 2).to(device)
    image = image_batch[0].to(device)
    conv_output = conv_layer(image)
    print("Convoluted output:")
    print(conv_output)
    print(conv_output.shape)
    max_pool_output = max_pool_layer(conv_output)
    print("Max pooled output:")
    print(max_pool_output)
    print(max_pool_output.shape)