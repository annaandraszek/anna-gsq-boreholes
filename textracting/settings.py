
report_local_path = 'reports/QDEX/'
model_path = 'models/'
result_path = 'results/'
dataset_path = 'datasets/'
production_path = 'production/'
toc_tree_model_file = model_path + 'toc_tree_model.pkl'
fig_tree_model_file = model_path + 'fig_tree_model.pkl'
headid_nb_model_file = model_path + 'cnb_head_id_model.pkl'
marginals_model_file_tree = model_path +'marginals_tree_model.pkl'
marginals_model_file_CNB = model_path +'marginals_CNB_model.pkl'
marginals_model_file_forest = model_path + 'marginals_forest_model_v2.pkl'
marginals_id_dataset = dataset_path + 'identified_marginals_dataset.csv'
marginals_id_trans_dataset = dataset_path + 'identified_trans_marginals_dataset.csv'
page_extraction_dataset = dataset_path + 'page_extraction_dataset.csv'
heading_id_intext_model_file = model_path + 'heading_intext_CNB_model.pkl'
heading_classification_model_file = model_path + 'heading_classification_CNB.pkl'


def get_s3_location(file_id, format='pdf'):
    return 'QDEX/' + file_id + '/' + get_report_name(file_id, file_extension=format)


def get_report_name(file_id, local_path=False, file_extension=None):
    file = ''
    if local_path:
        file = report_local_path + str(file_id) + '/'
    file += "cr_" + str(file_id) + "_1"
    if file_extension:
            file += file_extension
    return file


def get_file_from_training(folder, file_id, local_path, extension='.json'):
    file = ''
    if local_path:
        file = 'training/' + folder + '/'
    if not 'cr_' in str(file_id):
        if not extension in str(file_id):
            file += get_report_name(file_id)
            return file + "_" + folder + extension
    return file + str(file_id) + "_" + folder + extension


def get_restructpageinfo_file(file_id, local_path=True):
    return get_file_from_training('restructpageinfo', file_id, local_path)


def get_kvs_file(file_id, local_path=True):
    return get_file_from_training('kvs', file_id, local_path, extension='.csv')


def get_tables_file(file_id, local_path=True):
    return get_file_from_training('tables', file_id, local_path, extension='.csv')


def get_full_json_file(file_id, local_path=True):
    return get_file_from_training('fulljson', file_id, local_path)

