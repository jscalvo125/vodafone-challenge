# Use Python38
FROM python:3.8
# Copy requirements.txt to the docker image and install packages
COPY requirements.txt /
RUN pip install -r requirements.txt && \
    python -m spacy download en_core_web_sm && \
    python -m spacy download en_core_web_md
RUN pip install models/en_core_web_lg-2.3.1.tar.gz
# Set the WORKDIR to be the folder
COPY . .
# Expose port 8080
EXPOSE 8080
ENV PORT 8080
WORKDIR /
CMD ["python", "main.py"]