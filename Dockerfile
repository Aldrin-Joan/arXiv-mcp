FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir "numpy<2.0.0" && \
    pip install --no-cache-dir .

ENV ARXIV_KEEP_PDFS=false
ENV ARXIV_DOWNLOAD_DIR=/tmp/arxiv_mcp_downloads

EXPOSE 8000

ENTRYPOINT ["arxiv-mcp"]
