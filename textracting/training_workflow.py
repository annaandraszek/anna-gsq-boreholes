import toc_classification
import fig_classification
import heading_id_toc
import marginals_classification
import page_identification
import page_extraction
import heading_id_intext
import heading_classification
import settings
import glob
import re
import os


def save_dataset(df, name):
    path = settings.get_dataset_path(name)
    if not os.path.exists(settings.dataset_path + '/' + settings.dataset_version):
        os.mkdir(settings.dataset_path + '/' + settings.dataset_version)
    if not os.path.exists(path):
        df.to_csv(path, index=False)
    else:
        print('Dataset already exists here. To prevent overwriting annotation, delete it manually first.')


def create_training_sets(): # creating of training sets, up to their annotation
    toc_df = toc_classification.create_dataset()
    save_dataset(toc_df, 'toc')

    fig_df = fig_classification.create_dataset()
    save_dataset(fig_df, 'fig')

    heading_id_toc_df = heading_id_toc.create_identification_dataset()
    save_dataset(heading_id_toc_df, 'heading_id_toc')
    proc_df = heading_id_toc.pre_process_id_dataset(datafile=settings.get_dataset_path('heading_id_toc'))
    save_dataset(proc_df, 'processed_heading_id_toc')

    marginals_df = marginals_classification.create_dataset()
    save_dataset(marginals_df, 'marginal_lines')

    page_id_df = page_identification.create_dataset()
    save_dataset(page_id_df, 'page_id')

    page_ex_df = page_extraction.create_dataset()
    save_dataset(page_ex_df, 'page_extraction')

    heading_id_intext_df = heading_id_intext.create_dataset()
    save_dataset(heading_id_intext_df, 'heading_id_intext')

    heading_class_df = heading_classification.create_dataset()
    save_dataset(heading_class_df, 'heading_classification')


def train_models():  # run model training. output report of how they all did
    toc_classification.train()
    fig_classification.train()

    heading_id_toc_nn = heading_id_toc.NeuralNetwork()
    heading_id_toc_nn.train()

    marginals_classification.train()

    page_id_nn = page_identification.NeuralNetwork()
    page_id_nn.train()

    page_ex_nn = page_extraction.NeuralNetwork()
    page_ex_nn.train()

    heading_id_intext.train()
    heading_id_intext.train(model_file=settings.get_model_path('heading_id_intext_no_toc'))

    heading_classification.train()  # commenting this out now as I don't use it


if __name__ == "__main__":
    settings.dataset_version = 'expansion1'
    # add preliminary step that textracts and clean_and_restructs files
    # training files will be all in training/restructpageinfo folder, so should filter the members of that first

    create_training_sets()
    # manually annotate training sets
    train_models()
    # save output report