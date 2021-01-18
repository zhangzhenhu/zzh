#!/usr/bin/env bash

__script_dir=$(cd "$(dirname "$0")"; pwd)

project_home="${__script_dir}"


rm -fr "${project_home}/build/"

python3 "${project_home}/setup.py" build_ext --inplace

sphinx-apidoc -o "${project_home}/docs/source"  "${project_home}/zzh"
cd "${project_home}/docs/"
make clean
make html
cd -
