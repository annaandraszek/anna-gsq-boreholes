# gsq-boreholes

This is a repository of explorational machien learning work completed with the goal of extracting information from QDEX Reports. 

The main chunk of work is in /textracting. In that work, a Python workflow is used to pull reports from a DNRME AWS S3 bucket, gain their OCR information with AWS Textract, then use a number of machine learning classifiers to identify parts of the report with the goal of bookmarking its sections and exporting those sections to text.

While the Bookmarker has value as its own product, this was intended to be a pre-processing step prior to performing natural language processing for custom entity recognition to find boreholes in the reports.

## Instructions
- Need: DNRME AWS account with permission to access and write to S3 buckets and use Textract. AWS credentials and config files.
- Install dependencies in requirements.txt (todo: create requirements.txt)
- Configure bucket names, destination paths, etc in settings.py (todo: edit settings.py with more variables to allow this)
- In main of workflow.py, give the numerical IDs of the reports to run the Bookmarker on. Currently, a random sample of 20 is used. (todo: instead of editing workflow.py, make main accept args such as docids from command line)
- The output witll be a bookmarked PDF (and word document containing text of the sections)

## Performance
Depending on the similar of the reports to process to reports the program has been trained on, and whether your reports contain a Table of Contents, quality of results will vary. Reports without a Table of Contents will have much worse results than if they did, as the program relies on having the headings from it as a comparison against potential headings in the body of the report. Some headings may still be found correctly without a TOC, but this will be inconsistent.
