#!/bin/bash

INCL="${1}"
EXCL="${2}"

if [[ "${#}" == 3 ]]; then
  IFS='|' read -ra out <<< "$INCL"
  for i in "${!out[@]}"; do
    out[$i]="${out[$i]}.*${3}"
  done
  INCL="$(echo "${out[@]}" | tr " " "|")"
fi

if [ ! -z "$EXCL" -a "$EXCL" != " " ]; then
      dvc stage list | awk '{ print $1 }' | grep -E "$INCL" | grep -v "$EXCL" | sed 's/^\|$//g' | paste -sd" " -
else
  dvc stage list | awk '{ print $1 }' | grep -E "$INCL" | sed 's/^\|$//g' | paste -sd" " -
fi