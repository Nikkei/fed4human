#!/usr/bin/env bash

WORK_DIR=./data/wikinews/work/
OUT_DIR=./data/wikinews/

usage() {
  cat << _EOT_
Usage:
  scripts/build_synthetic_data.sh
    --work-dir=<WORK_DIR>
    --out-dir=<OUT_DIR>
*** NOTE: specify arguments with \"=\" ***

Options:
  --work-dir  path to working directory (default: ./data/wikinews/work)
  --out-dir   path to output directory (default: ./data/wikinews/)
_EOT_
}

while getopts h-: opt; do
  if [[ $opt = "-" ]]; then
    opt=$(echo "${OPTARG}" | awk -F "=" '{print $1}')
    OPTARG=$(echo "${OPTARG}" | awk -F "=" '{print $2}')
  fi

  case "$opt" in
  work-dir)
    WORK_DIR=$OPTARG
    ;;
  out-dir)
    OUT_DIR=$OPTARG
    ;;
  h | help)
    usage
    exit 0
    ;;
  *)
    echo "invalid option -- $opt"
    exit 1
    ;;
  esac
done

venvs/named_entity_and_kanji_misconversion/bin/python scripts/synthetic_data_generation/generate_synthetic_data.py \
  config/synthetic_data_generation.json \
  $WORK_DIR \
  --categories named_entity kanji_misconversion
venvs/remaining_categories/bin/python scripts/synthetic_data_generation/generate_synthetic_data.py \
  config/synthetic_data_generation.json \
  $WORK_DIR \
  --categories antonym numerical_value date digit unit

venvs/remaining_categories/bin/python scripts/synthetic_data_generation/filter_synthetic_data.py \
  $WORK_DIR \
  config/synthetic_data_generation.json \
  --category named_entity
venvs/remaining_categories/bin/python scripts/synthetic_data_generation/filter_synthetic_data.py \
  $WORK_DIR \
  config/synthetic_data_generation.json \
  --category antonym

venvs/remaining_categories/bin/python scripts/synthetic_data_generation/get_samples_and_few-shot_examples.py \
  $WORK_DIR \
  $OUT_DIR
