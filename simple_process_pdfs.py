# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 16:51:44 2025

@author: td00654
"""

from grobid_client.grobid_client import GrobidClient

#----------------PDF process-------------------
# Create client instance
client = GrobidClient(grobid_server="http://localhost:8070")

in_path = r"C:\Users\td00654\OneDrive - University of Surrey\Documents\EDRC LLM Project\CREDS Papers\downloads"
out_path = r"C:\Users\td00654\OneDrive - University of Surrey\Documents\EDRC LLM Project\CREDS Papers\CREDS - HTML"
# Process documents
client.process("processFulltextDocument", in_path, out_path, n=10)

