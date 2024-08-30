#FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel
RUN pip install --upgrade pip
RUN pip install --upgrade pip setuptools
# RUN apt-get install -y tensorrt

# BELONGS TO TORCH
RUN pip install torchsummary
RUN pip install torchvision 
RUN pip install tensorboard
RUN pip install torch-tb-profiler 

# SIDE PACKAGES
RUN pip install matplotlib
RUN pip install opencv-python
RUN pip install imgaug
RUN pip install typing-extensions --upgrade
RUN pip install tqdm
RUN pip install pandas

# Required in transformers - sometimes
RUN pip install einops

# If required, checkout https://github.com/NVIDIA/nvidia-container-toolkit/issues/258 for edge tpu compiler installation
# And https://github.com/mattiantonini/edgetpu-compiler-container/blob/master/Dockerfile

# RUN pip install --no-cache nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com

# MAKE OPENCV RUNNABLE
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

CMD [ "/bin/bash" ]