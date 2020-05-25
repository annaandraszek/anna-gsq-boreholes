How to use:
1.	Ensure config and credentials files for AWS are correctly configured (following https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html). 
2.	Create an anaconda environment from the environment.yml file (following https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or, since that environment will contain a lot of unnecessary packages for this sub-project, install libraries as needed by imports. 
3.	Edit variables in textsettings.py as appropriate to AWS account.
4.	Run textmain.py. If run without arguments, it defaults to getting a sample of 20 report IDs and trying to run Textract on all files in them (if appropriate). In the process of this, the files are also copied locally to reports/QDEX/[reportID]. Results from Textract are saved to textract_result/. The main result of Textract is json, then kvs (key value pairs) and tables are found in that response. You can ignore restructpageingo – it contains extra processing.

To run textmain.py for ALL reports (not just a sample) run it in the following way:
“python textmain.py –all”

To run textmain.py for SPECIFIC reports:
“python textmain.py –id [space separated reports ids]” eg. “python textmain.py –id 32487 2349 123 12309” 

