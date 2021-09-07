#!/bin/bash
app="vodafone.amazon.ml"
DOCKER_BUILDKIT=0 docker build -t ${app} .
docker run -d --name myvodafonecontainer -p 8080:8080 ${app}