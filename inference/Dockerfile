ARG FUNCTION_DIR="/function/"

FROM public.ecr.aws/docker/library/python:buster as build-image

RUN apt-get update && \
  apt-get install -y \
  g++ \
  make \
  cmake \
  unzip \
  libcurl4-openssl-dev

ARG FUNCTION_DIR
RUN mkdir -p ${FUNCTION_DIR}
COPY *.py ${FUNCTION_DIR}

RUN pip uninstall --yes jupyter
RUN pip install --target ${FUNCTION_DIR} torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --target ${FUNCTION_DIR} transformers==4.35.2
RUN pip install --target ${FUNCTION_DIR} scikit-learn
RUN pip install --target ${FUNCTION_DIR} awslambdaric 
RUN pip install --target ${FUNCTION_DIR} sentencepiece protobuf 

FROM public.ecr.aws/docker/library/python:buster

# Include global arg in this stage of the build
ARG FUNCTION_DIR
# Set working directory to function root directory
WORKDIR ${FUNCTION_DIR}

# Copy in the built dependencies
COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}

ENTRYPOINT [ "python3", "-m", "awslambdaric" ]

# This will get replaced by the proper handler by the CDK script
CMD [ "oads.handler" ]
