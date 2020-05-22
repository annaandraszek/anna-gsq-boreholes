## @package textractor
#@file
read_bucket = 'gsq-horizon'  # bucket for reading reports
write_bucket = 'gsq-ml2'  # bucket for writing reports (because do not have permissions to write to read bucket rn)
#region = 'ap-southeast-2'  # AWS region: Asia Pacific (Sydney) # don't need to code this if in config
num_sample = 20  # size of the report sample
all_files = True  # True if to textract all documents in a report; False if just the first one (_1)
