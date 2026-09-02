#!/usr/bin/env bash

uv python pin 3.10.15

uv venv venvs/preprocessing
source venvs/preprocessing/bin/activate
uv pip install -r requirements/preprocessing.txt
deactivate

uv venv venvs/named_entity_and_kanji_misconversion
source venvs/named_entity_and_kanji_misconversion/bin/activate
uv pip install -r requirements/named_entity_and_kanji_misconversion.txt
deactivate

uv venv venvs/remaining_categories
source venvs/remaining_categories/bin/activate
uv pip install -r requirements/remaining_categories.txt
deactivate

uv venv venvs/training
source venvs/training/bin/activate
uv pip install -r requirements/training.txt
deactivate

uv venv venvs/evaluation
source venvs/evaluation/bin/activate
uv pip install -r requirements/evaluation.txt
deactivate
