#!/bin/bash

# make a virtual env 
virtualenv -p python3 text-anno-tool
source text-anno-tool/bin/activate
pip install -r requirements.txt

python manage.py runserver

echo "Server is running... "
