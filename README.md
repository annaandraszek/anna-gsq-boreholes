# gsq-boreholes

This is a repository of explorational machine learning work completed with the goal of extracting information from QDEX Reports. 

The main chunk of work is in textracting/. In that work, a Python workflow is used to pull reports from a DNRME AWS S3 bucket, gain their OCR information with AWS Textract, then use a number of machine learning classifiers to identify parts of the report with the goal of bookmarking its sections and exporting those sections to text.

While the Bookmarker has value as its own product, this was intended to be a pre-processing step prior to performing natural language processing for custom entity recognition to find boreholes in the reports.

There is also work on borehole reference extraction in textracting/borehole. 

## Instructions
- Need: AWS account with permission to access and write to S3 buckets and use Textract. AWS credentials and config files.
- Configure environment using environment.yaml (anaconda enviornment file, see: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
- Configure bucket names, destination paths, etc in settings.py (todo: edit settings.py with more variables to allow this)
- See readme inside textracting/ for information in execution. 
- The output will be a bookmarked PDF (and word document containing text of the sections)

## Performance
Depending on the similar of the reports to process to reports the program has been trained on, and whether your reports contain a Table of Contents, quality of results will vary. Reports without a Table of Contents will have much worse results than if they did, as the program relies on having the headings from it as a comparison against potential headings in the body of the report. Some headings may still be found correctly without a TOC, but this will be inconsistent.
