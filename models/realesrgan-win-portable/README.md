## Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data

# Windows Portable Files (.exe, .dll, .bin, .param)

https://github.com/xinntao/Real-ESRGAN/tree/master?tab=readme-ov-file#portable-executable-files-ncnn

This executable file is **portable** and includes all the binaries and models required. No CUDA or PyTorch environment is needed.

Note that it may introduce block inconsistency (and also generate slightly different results from the PyTorch implementation), because this executable file first crops the input image into several tiles, and then processes them separately, finally stitches together.

This executable file is based on the wonderful [Tencent/ncnn](https://github.com/Tencent/ncnn) and [realsr-ncnn-vulkan](https://github.com/nihui/realsr-ncnn-vulkan) by [nihui](https://github.com/nihui).

**Usage:** https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan#computer-usages

