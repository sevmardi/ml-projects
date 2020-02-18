#!/bin/bash

# make a virtual env
virtualenv -p python3 text-summarizer
source text-summarizer/bin/activate
pip install -r requirements.txt

python manage.py runserver

echo "Server is running... "
