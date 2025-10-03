# -*- coding: utf-8 -*-
"""
This script processes a directory of PDF files using the GROBID service.

It initializes a GROBID client, points it to a local GROBID server instance,
and processes all PDF documents found in the specified input directory. The
output is structured TEI XML, which is saved to the specified output directory.

Note:
    This script requires a running GROBID server. The server's address
    should be configured in the `GrobidClient` instantiation.
"""

from grobid_client.grobid_client import GrobidClient

#----------------PDF process-------------------
# Create client instance
client = GrobidClient(grobid_server="http://localhost:8070")

in_path = r"C:\Users\td00654\OneDrive - University of Surrey\Documents\EDRC LLM Project\CREDS Papers\downloads"
out_path = r"C:\Users\td00654\OneDrive - University of Surrey\Documents\EDRC LLM Project\CREDS Papers\CREDS - HTML"
# Process documents
client.process("processFulltextDocument", in_path, out_path, n=10)

