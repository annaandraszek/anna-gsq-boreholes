## @file
# Settings

import glob

report_local_path = '../reports/QDEX/'
test_local_path = '../reports/test/'
model_path = 'models/'
result_path = 'results/'
dataset_path = 'datasets/'
production_path = 'production/'
toc_tree_model_file = model_path + 'toc_tree_model.pkl'
fig_tree_model_file = model_path + 'fig_tree_model.pkl'
#headid_nb_model_file = model_path + 'cnb_head_id_model.pkl'
#marginals_model_file_tree = model_path +'marginals_tree_model.pkl'
#marginals_model_file_CNB = model_path +'marginals_CNB_model.pkl'
marginals_model_file_forest = model_path + 'marginals_forest_model_v2.pkl'

heading_id_intext_model_file = model_path + 'heading_intext_CNB_model.pkl'
heading_id_intext_model_file_no_toc = model_path + 'heading_intext_CNB_no_toc_model.pkl'

heading_classification_model_file = model_path + 'heading_classification_CNB.pkl'

marginals_id_dataset = dataset_path + 'identified_marginals_dataset.csv'
marginals_id_trans_dataset = dataset_path + 'identified_trans_marginals_dataset.csv'
page_extraction_dataset = dataset_path + 'page_extraction_dataset.csv'


ml = {'toc': 'toc',
      'fig': 'fig',
      'head_id_toc': 'heading_id_toc',
      'head_id_intext': 'heading_id_intext',
      'head_id_intext_no_toc': None,
      'proc_head_id_toc': None,
      'marginals': 'marginal_lines',
      'page_id': 'page_id',
      'page_ex': 'page_extraction',
      'head_class': ' heading_classification'}
ml['proc_head_id_toc'] = 'processed_' + ml['head_id_toc']
ml['head_id_intext_no_toc'] = ml['head_id_intext'] + '_no_toc'


dataset_version = 'expansion1'  # folder of models/data in progress of being developed
production = 'production'  # folder of last working models


def get_model_path(model, mode=dataset_version, tokeniser=False, classes=False):
    if 'boreholes' in mode:
        path = ''
    else:
        path = '../'
    path += model_path + mode + '/' + model
    # if type:
    #     path += '_' + type
    path += "_model"
    #if ('heading_id_toc' or 'page') in model and not tokeniser:
        #path += ".h5"
    if tokeniser:
        path += '_tokeniser.joblib'
    elif classes:
        path += '_class_dict.joblib'
    else:
        path += '.pkl'
    return path


def get_report_page_path(report_id, page):
    rep = get_report_name(report_id, local_path=True, file_extension=".pdf")
    page_file = rep.split('.pdf')[0]
    page_file += '_page' + str(page) + '.png'
    return page_file


def get_dataset_path(dataset, training_name=dataset_version):
    if 'boreholes' in dataset_version:
        return dataset_path + training_name + '/' + dataset + '_dataset.csv'
    else:
        return '../' + dataset_path + training_name + '/' + dataset + '_dataset.csv'


def get_s3_location(report_id, format='pdf', file_num=1):
    return 'QDEX/' + report_id + '/' + get_report_name(report_id, file_extension=format, file_num=file_num)


def get_s3_subdir(docid):
    return 'QDEX/' + docid + '/'


def get_report_name(report_id, local_path=False, file_extension=None, file_num=1):
    file = ''
    if local_path:
        if isinstance(local_path, str) and 'test' in local_path:
            file = test_local_path
        else:
            file = report_local_path
        file += str(report_id) + '/'
    file += "cr_" + str(report_id) + "_" + str(file_num)
    if file_extension:
            file += file_extension
    return file


def get_file_from_training(folder, report_id, local_path, extension='.json', training=True, file_num=1, extrafolder=None):
    file = ''
    if local_path:
        fullfolder = folder
        if extrafolder:
            fullfolder = extrafolder + '/' + folder
        if training:
            file = '../training/' + fullfolder + '/'
        else:
            file = 'C:\\Users\\andraszeka\\OneDrive - ITP (Queensland Government)\\textract_result/' + fullfolder + '/'
    if not 'cr_' in str(report_id):
        if not extension in str(report_id):
            file += get_report_name(report_id, file_num=file_num)
            return file + "_" + folder + extension
    return file + str(report_id) + "_" + folder + extension


def get_bookmarked_file(report_id, test=False, test_i='', filenum=1):
    path = report_local_path
    if test:
        path = test_local_path
        files = path + str(report_id) + '/*'
        if not test_i:
            fpaths = glob.glob(files)
            test_i = len(fpaths)
    file = path + report_id + "/cr_" + report_id + "_" + filenum + "_" + str(test_i) + "_bookmarked.pdf"
    return file


def get_restructpageinfo_file(report_id, local_path=True, training=True, file_num=1, extrafolder=None):
    return get_file_from_training('restructpageinfo', report_id, local_path, training=training, file_num=file_num, extrafolder=extrafolder)


def get_text_file(report_id, local_path=True, training=True, file_num=1, extrafolder=None):
    return get_file_from_training('text', report_id, local_path, training=training, file_num=file_num, extension='.txt', extrafolder=extrafolder)


def get_kvs_file(report_id, local_path=True, training=True, file_num=1, extrafolder=None):
    return get_file_from_training('kvs', report_id, local_path, extension='.csv', training=training, file_num=file_num, extrafolder=extrafolder)


def get_tables_file(report_id, local_path=True, training=True, file_num=1, bh=False, extrafolder=None):
    s = 'tables'
    if bh:
        s += '_bh'
    return get_file_from_training(s, report_id, local_path, extension='.csv', training=training, file_num=file_num, extrafolder=extrafolder)


def get_full_json_file(report_id, local_path=True, training=True, file_num=1, extrafolder=None):
    return get_file_from_training('fulljson', report_id, local_path, training=training, file_num=file_num, extrafolder=extrafolder)


def pad_num(num):
    if len(str(num)) == 2:
        num = '0' + str(num)
    elif len(str(num)) == 1:
        num = '00' + str(num)
    return num


def get_word_file(docid, file_num, service):
    base_path = path = 'C:\\Users\\andraszeka\OneDrive - ITP (Queensland Government)\\textract_result\\qutsample'
    id_path = base_path + '/' + service + '/texts/' + str(docid) +'/'
    fname = str(docid) + '_' + str(pad_num(file_num)) + '.docx'
    return id_path + fname