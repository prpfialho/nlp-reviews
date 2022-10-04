import os, random, pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from tmtoolkit.corpus import Corpus, tokens_table, lemmatize, to_lowercase, dtm, remove_punctuation, \
    numbers_to_magnitudes, filter_clean_tokens, corpus_collocations, join_collocations_by_statistic, \
    remove_common_tokens, remove_uncommon_tokens, filter_for_pos
from tmtoolkit.bow.bow_stats import tfidf, sorted_terms_table
from tmtoolkit.topicmod.visualize import plot_doc_topic_heatmap
from tmtoolkit.topicmod.tm_lda import compute_models_parallel
from tmtoolkit.topicmod.model_io import print_ldamodel_topic_words
from tmtoolkit.topicmod.tm_lda import evaluate_topic_models
from tmtoolkit.topicmod.evaluate import results_by_parameter
from tmtoolkit.topicmod.visualize import plot_eval_results
from tmtoolkit.tokenseq import pmi3
from tmtoolkit.corpus import print_summary
from tmtoolkit.topicmod.model_stats import topic_word_relevance
from tmtoolkit.topicmod.visualize import generate_wordclouds_for_topic_words
from tmtoolkit.corpus import save_corpus_to_picklefile, load_corpus_from_picklefile
from tmtoolkit.bow.bow_stats import doc_lengths
from tmtoolkit.topicmod.model_stats import generate_topic_labels_from_top_words
from tmtoolkit.topicmod.model_stats import most_relevant_words_for_topic, \
    least_relevant_words_for_topic
from tmtoolkit.topicmod.model_io import ldamodel_top_topic_words, ldamodel_top_word_topics, \
    ldamodel_top_doc_topics, ldamodel_top_topic_docs
from tmtoolkit.topicmod.model_io import save_ldamodel_summary_to_excel
from tmtoolkit.topicmod.model_stats import exclude_topics


#
#     topic
#       modelling
#           methods
#
def reviews_tocsv(revtxt_inp='/home/pfialho/data/reviews.txt', revcsv_outp="/home/pfialho/data/reviews.csv", load_fromcache=1):
    print('.. get csv')
    if load_fromcache and os.path.isfile(revcsv_outp):
        return revcsv_outp

    reviews_csv = ""
    with open(revtxt_inp) as f:
        i = 0
        for line in f:
            reviews_csv += str(i) + ',"' + line.replace('"', '""').strip() + '"\n'
            i += 1

    with open(revcsv_outp, "w") as text_file:
        text_file.write(reviews_csv)

    return revcsv_outp


# TODO: generate various corpora versions,
#  with different normalization techniques,
#  to reduce/increase the number of available tokens
def tm_revcsv_normcorpus(revcsv_inp="/home/pfialho/data/reviews.csv", revcorp_outp='corp_norm.p', load_fromcache=1):
    print('.. get corpus: lemma punct lower rem-st3 rem-num rem-comm0.85 rem-uncomm0.05 filter-posN')
    if load_fromcache and os.path.isfile(revcorp_outp):
        return load_corpus_from_picklefile(revcorp_outp)

    corp = Corpus.from_tabular(revcsv_inp, language='en',
                               id_column=0, text_column=1, max_workers=6)
    print_summary(corp)
    # toktbl = tokens_table(corp)
    lemmatize(corp)
    remove_punctuation(corp)
    to_lowercase(corp)
    # numbers_to_magnitudes(corp)
    filter_clean_tokens(corp, remove_shorter_than=3, remove_numbers=True)
    remove_common_tokens(corp, df_threshold=0.85)
    remove_uncommon_tokens(corp, df_threshold=0.05)
    filter_for_pos(corp, 'N')

    save_corpus_to_picklefile(corp, revcorp_outp)
    print_summary(corp)

    # colloc = join_collocations_by_statistic(corp, statistic=pmi3, threshold=-8.25, return_joint_tokens=True)
    return corp


def tm_getdtm(revcorpora=None, load_fromcache=1):
    dtm_outp = "mat_docs_topics.p"
    if load_fromcache and os.path.isfile(dtm_outp):
        return pickle.load(open(dtm_outp, "rb"))

    dtmat, doc_labels, vocab = dtm(revcorpora, return_doc_labels=True, return_vocab=True)
    pickle.dump((dtmat, doc_labels, vocab), open(dtm_outp, "wb"))

    return dtmat, doc_labels, vocab


def tm_evalmodels(dtmat, load_fromcache=1, showplot=0):
    eval_outp = "eval_results_by_topics.p"
    if load_fromcache and os.path.isfile(eval_outp):
        return pickle.load(open(eval_outp, "rb"))
    else:
        var_params = [{'n_topics': k, 'alpha': 1 / k}
                      for k in range(20, 121, 10)]

        const_params = {
            'n_iter': 1000,
            'random_state': 20191122,  # to make results reproducible
            'eta': 0.1,  # sometimes also called "beta"
        }

        eval_results = evaluate_topic_models(dtmat,
                                             varying_parameters=var_params,
                                             constant_parameters=const_params,
                                             coherence_mimno_2011_top_n=10,
                                             coherence_mimno_2011_include_prob=True,
                                             return_models=True)
        eval_results_by_topics = results_by_parameter(eval_results, 'n_topics')
        pickle.dump(eval_results_by_topics, open(eval_outp, "wb"))

    if showplot:
        plot_eval_results(eval_results_by_topics)
        plt.show()

    return eval_results_by_topics


# run tm_evalmodels first, to select the ideal number of topics (num_topics)
def tm_getmodel(eval_results_by_topics, num_topics, dtmat, vocab):
    best_tm = [m for k, m in eval_results_by_topics if k == num_topics][0]['model']

    doc_lengths_bg = doc_lengths(dtmat)
    topic_labels = generate_topic_labels_from_top_words(
        best_tm.topic_word_,
        best_tm.doc_topic_,
        doc_lengths_bg,
        np.array(vocab),
        lambda_=0.6
    )

    return best_tm, topic_labels


def tm_filter_exp(best_tm, topic_labels, uninform_topics=[0, 2, 8, 9, 10, 11, 15, 19], toxlsx_path=''):
    new_doc_topic, new_topic_word, new_topic_mapping = \
        exclude_topics(uninform_topics, best_tm.doc_topic_,
                       best_tm.topic_word_, return_new_topic_mapping=True)

    new_topic_labels = np.delete(topic_labels, uninform_topics)

    # topic_word_rel = topic_word_relevance(best_tm.topic_word_, best_tm.doc_topic_,
    #                                       doc_lengths_bg, lambda_=0.6)

    if toxlsx_path:
        dtmat, doc_labels, vocab = tm_getdtm()
        sheets = save_ldamodel_summary_to_excel(toxlsx_path,
                                                new_topic_word, new_doc_topic,
                                                doc_labels, vocab,
                                                dtm=dtmat,
                                                topic_labels=new_topic_labels)

    return new_doc_topic, new_topic_labels


if __name__ == '__main__':
    print(". build document term matrix")
    docterm_mtx, doc_labels, rawtopics = tm_getdtm(tm_revcsv_normcorpus(revcsv_inp=reviews_tocsv()))

    print(". eval topic models")
    # evald_models = tm_evalmodels(docterm_mtx, showplot=1)
    # sys.exit(1)
    evald_models = tm_evalmodels(docterm_mtx, showplot=0)

    print(". select topic model")
    topcmodl, topclbl = tm_getmodel(evald_models, 30, docterm_mtx, rawtopics)
    topcmodl_filt, topclbl_filt = tm_filter_exp(topcmodl, topclbl)

    print(". plot topic model subset")
    fig, ax = plt.subplots(figsize=(32, 8))
    which_docs = random.sample(doc_labels, 5)
    plot_doc_topic_heatmap(fig, ax, topcmodl_filt, doc_labels,
                           topic_labels=topclbl_filt,
                           which_documents=which_docs)
    plt.show()


