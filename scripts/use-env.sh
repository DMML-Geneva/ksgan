#!/bin/bash

if [[ "${#}" == 1 ]]; then
	REQUIREMENTS="${1}"
else
	REQUIREMENTS="./requirements.txt"
fi

find_in_conda_env(){
    conda env list | grep "${@}" >/dev/null 2>/dev/null
}


if ! find_in_conda_env "ksgan"; then
  echo "Creating conda env"
  conda create -y --name ksgan python=3.11
  conda activate ksgan
  conda install -y -c conda-forge texlive-core
  conda install -y pip
  pip install --upgrade pip
  pip install wheel==0.38.4
  pip install $(grep '^torch' $REQUIREMENTS) --index-url https://download.pytorch.org/whl/cu121
  pip install $(grep '^matplotlib' $REQUIREMENTS)
fi

conda activate ksgan

./scripts/install-deps.sh $REQUIREMENTS
source ./scripts/read-env-file.sh
