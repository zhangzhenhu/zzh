#!/usr/bin/env bash
__script_dir=$(cd "$(dirname "$0")"; pwd)


python3 ${__script_dir}/setup.py build_ext --inplace