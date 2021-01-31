FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04


ARG PYTHON_VERSION=3.8

ARG CUDA=11.1
ARG LIB_DIR_PREFIX=x86_64
ARG LIBNVINFER=7.2.2-1
ARG LIBNVINFER_MAJOR_VERSION=7

# Needed for string substitution
SHELL ["/bin/bash", "-c"]

# Install necessary building tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         git \
         curl \
         sudo \
         bzip2 \
         libx11-6 \
         ca-certificates \
         wget \
         libjpeg-dev \
         libpng-dev \
         software-properties-common \
         unzip && \
     rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt-get update && apt-get -y dist-upgrade && apt-get purge -y libboost-all-dev && \
    apt-get install -f -y libboost-all-dev && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt-get update
RUN apt-get install -y vim lshw && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# Install TensorRT if not building for PowerPC
RUN [[ "${ARCH}" = "ppc64le" ]] || { apt-get update && \
        apt-get install -y --no-install-recommends libnvinfer${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
        libnvinfer-plugin${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*; }


# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH


# Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# dynamic linker run-time bindings
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig

ENV LANG C.UTF-8


# Install conda
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=$PYTHON_VERSION && \
    /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH

RUN /opt/conda/bin/pip install --upgrade pip


# Install tensorflow using pip
RUN /opt/conda/bin/pip --no-cache-dir install --upgrade tensorflow

# Install tf dataset and opencv
RUN /opt/conda/bin/pip --no-cache-dir install --upgrade \
    # tensorflow_datasets \
    opencv-python

# Install library for cv2
RUN apt update && apt install -y libgl1-mesa-glx

# Make link to libcusolver.so.10, since it's required when registering GPU devices
RUN ln /usr/local/cuda/lib64/libcusolver.so.11 /usr/local/cuda/lib64/libcusolver.so.10

# Some TF tools expect a "python" binary
RUN ln -s $(which python3.8) /usr/local/bin/python


# Install visualization packages
RUN /opt/conda/bin/pip --no-cache-dir install --upgrade \
    seaborn \
    matplotlib

# Install packages for image processing
RUN /opt/conda/bin/pip --no-cache-dir install --upgrade \
    # numpy \
    pylibjpeg \
    pylibjpeg-libjpeg \
    pylibjpeg-openjpeg \
    pydicom \
    pillow
# RUN /opt/conda/bin/conda install gdcm -y -c conda-forge


# Install Jupyter
RUN /opt/conda/bin/conda install -y -c conda-forge \
    jupyterlab \
    jupyter_http_over_ws \
    ipywidgets \
    nodejs==15.3.0 && \
    /opt/conda/bin/conda clean -ya

RUN /opt/conda/bin/conda install -y -c anaconda nbformat && \
    /opt/conda/bin/conda install -y ipykernel && \
    /opt/conda/bin/conda clean -ya
RUN jupyter serverextension enable --py jupyter_http_over_ws

# RUN jupyter labextension install @telamonian/theme-darcula  # Not available for JupyterLab3.0 yet
RUN jupyter labextension install @arbennett/base16-monokai
RUN python3 -m ipykernel.kernelspec


# Create a working directory
RUN mkdir /app
WORKDIR /app
