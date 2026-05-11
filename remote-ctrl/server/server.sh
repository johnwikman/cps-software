#!/usr/bin/env bash

source ~/.bashrc

#pip3 install keyboard sb3_contrib

RELDIR="$( cd "$(dirname "$BASH_SOURCE")" >/dev/null 2>&1 ; pwd -P )"

exec "${RELDIR}/server_spider.py"
