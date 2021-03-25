# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 18:18:49 2021

@author: Anup w
"""

import requests

url = 'http://localhost:8501//predict_api'
r = requests.post(url,json={'caller_id':2403, 'open_by':397, 'loc':165,'category':215})

print(r.json())