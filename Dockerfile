ARG WORKER_CUDA_VERSION=11.8.0
FROM runpod/base:0.6.1-cuda${WORKER_CUDA_VERSION}
# Python dependencies
COPY requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

RUN mkdir /cache
#ENV SENTENCE_TRANSFORMERS_HOME=/runpod-volume
ENV SENTENCE_TRANSFORMERS_HOME=/cache

ADD handler.py .

# test run to prime model cache
RUN python3.11 handler.py

CMD python3.11 -u /handler.py
