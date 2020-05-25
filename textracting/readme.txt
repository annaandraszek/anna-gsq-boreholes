# workflow.py: Main file for pipeline of extracting text/segmenting/bookmarking report from the command line. This is the only file you'll need to run to do this.
# IF YOU'VE ALREADY RUN TEXTRACT ON YOUR FILES and want to avoid re-running, place your files in: (in this directory)
#	trainingFiles/
#		fulljson/ # full json response from textract. only essential component
#			cr_167_1_fulljson.json  # example of how files in this folder should be named. this is dictated by get_full_json_file() in paths.py.
#		restructpageinfo/  # optional, if you've already got this. quick and costless to create these though
#		tables/ # ok to not have these for this workflow, but need for borehole extraction
# You should also include original pdf/tif versions of reports in downloadedReports/, as these will be needed as a basis for bookmarking the report. Include these in the structure:
# 	downloadedReports/
#		[reportID]/  # eg. 167/
#			cr_167_1.pdf
#			cr_167_1.tif  # original was tif, conversion to pdf is saved too
#
# How to use:
#   If using AWS Textract: Modify variables in textractor/textsettings.py to match AWS config
#   Set up yoru environemtn: Install all needed libraries with environment.yml file
#   Run from command line:
#   For list of arguments:
#	python workflow.py -h
#   To run on a random sample of reports: (with default variable values for sampling)
#       python workflow.py
#   To run on specific report ids, use --id and space separated IDs, eg:
#       python workflow.py --id 2646 78932 32424
#
# In successfully running all parts of this process, the following directories/files are craeted: (in approximate order of creation)
# 	downloadedReports/[reportID]/[all original files that have been processed, and conversions to pdf is tif)
#	trainingFiles/fulljson/[json response files for all files processed]
#	trainingFiles/restructpageinfo/[""]
#	trainingFiles/tables/[""]
#	downloadedReports/[reportID]/cr_[reportID]_[fileNum]__bookmarked.pdf  # report with pdf bookmarks, according to section segmentation
#	downloadedReports/[reportID]/cr_[reportID]_[fileNum]_sections.docx  # report text, split according to section segmentation
#	downloadedReports/[reportID]/cr_[reportID]_[fileNum].json  # member data of this files's report.Report class, saved to json 
#
# To avoid duplicating work, workflow.py will look for this files to determine if it needs to do the processing to create them. If a file doesn't exist, only then will it create it - it will not overwrite. If all files exist, up to the __bookmarked file, no new processing will be done and a message will be returned that the file is already bookmarked (note that sections.docx and Report json files don't need to exist for this - delete the bookmarked.pdf file if these are missing and you wish to create them).
# This is also a good guide for locations if you find processes are running unecessarily anyway or if you have any issue with file locations. All directories are given relative to the directory of the readme. 
#
# HTML documentation can be automatically generated with Doxygen installed and running the command "doxygen Doxyfile" in this directory (Doxyfile is the config file). See http://www.doxygen.nl/manual/install.html. Once generated, go into the html folder and open index.html to navigate documentation for classes and files.
#
# I have endeavored to make this accurate and clear, but for enquiries, you can contact me at anna.k.andraszek@gmail.com.  