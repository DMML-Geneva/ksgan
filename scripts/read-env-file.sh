#!/bin/bash

set -o allexport
if test -f .env; then
  source .env
fi
set +o allexport
