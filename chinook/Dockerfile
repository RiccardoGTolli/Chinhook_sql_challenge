FROM python:3.9-slim-buster as production

USER root

ARG UID=1001
ARG GID=1002

ENV UID=1001
ENV GID=1002

# Install some basic packages including SQLite development headers
RUN apt-get update && apt-get install -y \
    gcc python3-dev \
    git 

ENV PYTHONPATH="${PYTHONPATH}:/chinook/"

COPY ./chinook/modules /chinook/modules
COPY ./chinook/output /chinook/output
COPY ./chinook/requirements.txt /chinook/requirements.txt 
COPY ./chinook/notebooks /chinook/notebooks

RUN pip install --upgrade pip
RUN pip install -r /chinook/requirements.txt
 
COPY ./entry_scripts/production.sh /chinook/entry_scripts/production.sh
RUN chmod +x -R /chinook/entry_scripts


ENTRYPOINT ["/bin/sh", "-c", "/chinook/entry_scripts/production.sh"]

# Multi-stage build
FROM production as development

# Show service name in bash prompt
RUN echo 'PS1="\e[0;31m[chinook]\e[m $PS1"' >> /root/.bashrc

RUN mkdir /chinook/static_test

COPY ./chinook/requirements_dev.txt /chinook/requirements_dev.txt
COPY ./static_test /chinook/static_test

RUN pip install -r /chinook/requirements_dev.txt
COPY ./entry_scripts/development.sh /chinook/entry_scripts/development.sh
RUN chmod +x -R /chinook/entry_scripts

# Copy the .bash_aliases file into the container
COPY ./static_test/bash_aliases.txt /root/.bash_aliases
# Append the sourcing command to the .bashrc file (to find the aliases)
RUN echo "source ~/.bash_aliases" >> /root/.bashrc

EXPOSE 8888
ENTRYPOINT ["/bin/sh", "-c", "/chinook/entry_scripts/development.sh"]
