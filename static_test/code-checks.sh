#!/bin/bash

LINE_LENGTH=120
HIGHLIGHT='\033[0;33m'
TEXT_RED="\e[31m"
NC='\033[0m' # No Color

DIRS=(modules/) # Directories to apply static checks to

for DIR in "${DIRS[@]}"; do # loop over the DIRS

    echo -e "${HIGHLIGHT}Running black for ${DIR}${NC}"
    black $DIR -l $LINE_LENGTH --check

    if [ $? -ne 0 ]; then
        echo -e "${TEXT_RED} black needs to be run before submitting a PR ${NC}"
        exit 1;
    fi;

    echo -e "${HIGHLIGHT}Running flake8 for ${DIR}${NC}"
    flake8 $DIR --exclude /logs/logging.py --ignore=W503,E203,E402 --max-line-length=$LINE_LENGTH --builtins="_"

    if [ $? -ne 0 ]; then
        echo -e "${TEXT_RED}flake8 errors must be resolved before submitting a PR${NC}"
        exit 1;
    fi;

    echo -e "${HIGHLIGHT}Running isort for ${DIR}${NC}"
    isort $DIR --profile black --skip notebooks --check-only 

    if [ $? -ne 0 ]; then
        echo -e "${TEXT_RED}isort needs to be run before submitting a PR ${NC}"
        exit 1;
    fi;

    echo -e "${HIGHLIGHT}Running mypy for ${DIR}${NC}"
    mypy $DIR --config-file static_test/mypy.ini || true

    if [ $? -ne 0 ]; then
        echo -e "${TEXT_RED}MyPy errors must be resolved for code check to pass ${NC}"
        exit 1;
    fi;
done