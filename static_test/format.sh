#!/bin/bash

LINE_LENGTH=120

DIRS=(modules/)

for DIR in "${DIRS[@]}"; do
    isort --profile black $DIR
    black -l $LINE_LENGTH $DIR
done


