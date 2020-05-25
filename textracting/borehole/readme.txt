The contents of this folder concern extraction of borehole information.

tables.py: classification of tables as containing borehole references (or not) and helper functions. This can be used to narrow down the tables from which boreholes are to be extracted.
extraction.py: extraction of borehole names and locations from tables. See Borehole Extraction Demo.ipynb in directory above for instructions on executing and documentation/Process of extracting boreholes from tables.pptx for more information.
well_card.py: barely started, this file was going to be for pulling information out of well card from the text (not tables).

Code for initially getting tables from Textract is in textractor/. Code for getting tables from MS Word files is in ../Comparing table extraction between different services.ipynb.