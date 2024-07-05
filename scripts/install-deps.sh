#!/bin/bash

if [[ "${#}" == 1 ]]; then
	REQUIREMENTS="${1}"
else
  SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
	REQUIREMENTS="$(dirname "$SCRIPT_DIR")/requirements.txt"
fi

pip install wheel==0.38.4
pip install $(grep 'torch==' $REQUIREMENTS) $(grep 'matplotlib==' $REQUIREMENTS)
pip install -r $REQUIREMENTS
pip install pdbpp ipdb
