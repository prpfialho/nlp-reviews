import pickle
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import coreferee
from wordcloud import WordCloud
import matplotlib.pyplot as plt


nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('spacytextblob')
nlp.add_pipe('coreferee')


def showwordcloud(adict):
    if len(adict) > 0:
        wc = WordCloud(width=1000, height=500)
        wc.generate_from_frequencies(adict)
        plt.figure(figsize=(15, 8))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.show()
    else:
        print('!!!!!!!!!!!!!! empty freq dict !!!!!!!!!!!!!!!!')


def filter_freqdicts(dict_freq, rem_lessthan, setofvals=0):
    dict_freq_filt = {}
    for k, v in dict_freq.items():
        v1 = v
        if setofvals:
            v1 = len(v)
        if v1 < rem_lessthan:
            continue

        dict_freq_filt[k] = v1

    return dict_freq_filt


def get_freqdicts(load_cache=1):
    if load_cache:
        return pickle.load(open('fr_nc-c-p-ne.p', "rb"))

    nchunk_revs = {}
    nchunk_freq = {}
    coref_freq = {}
    coref_revs = {}
    pos_freq = {}
    pos_revs = {}
    neg_freq = {}
    neg_revs = {}
    with open('reviews.txt') as f:
        i = 0
        for line in f:
            # if i > 50:
            #     break

            print(i, end=', ')
            doc = nlp(line)

            #
            #   Coreference resolution
            #
            # global frequency
            # and
            # review frequency
            #
            # TODO: replace with call to neuralcoref server
            #
            for chain in doc._.coref_chains.chains:
                for ment in chain.mentions:
                    ment1 = ment[0]
                    msolv = doc._.coref_chains.resolve(doc[ment1])

                    if msolv:
                        for msol1 in msolv:
                            msol = msol1.text
                            if msol.lower() in nlp.Defaults.stop_words:
                                continue

                            if not msol in coref_revs:
                                coref_revs[msol] = set()
                            coref_revs[msol].add(i)

                            val = coref_freq.get(msol, 0)
                            coref_freq[msol] = val + 1

            # sentiment analysis
            for sass in doc._.blob.sentiment_assessments.assessments:
                sass_w = " ".join(sass[0]).lower()
                if sass[1] > 0.5:
                    if not sass_w in pos_revs:
                        pos_revs[sass_w] = set()
                    pos_revs[sass_w].add(i)

                    val = pos_freq.get(sass_w, 0)
                    pos_freq[sass_w] = val + 1
                    # pos.append(' '.join(sass[0]))
                elif sass[1] < -0.5:
                    if not sass_w in neg_revs:
                        neg_revs[sass_w] = set()
                    neg_revs[sass_w].add(i)

                    val = neg_freq.get(sass_w, 0)
                    neg_freq[sass_w] = val + 1
                # else:     # neutral

                # if sass[1] > 0.3 or sass[1] < -0.3:
                #     sass_w = " ".join(sass[0]).lower()
                #
                #     if not sass_w in pol_revs:
                #         pol_revs[sass_w] = set()
                #     pol_revs[sass_w].add(i)
                #
                #     val = pol_freq.get(sass_w, 0)
                #     pol_freq[sass_w] = val + 1

            for sent in doc.sents:
                # print(sent.text)

                # for token in sent:
                #     if token.dep_ == 'nsubj':
                #         print(token.text, ' | ', token.dep_, ' | ', token.head.text, ' | ', token.head.pos_, ' | ',
                #               [child for child in token.children])

                #
                #   Syntax parsing : noun phrases
                #
                # global frequency
                # and
                # review frequency
                #
                for chunk in sent.noun_chunks:
                    # print(chunk.text, ' | ', chunk.root.text, ' | ', chunk.root.dep_, ' | ',
                    #       chunk.root.head.text, ' | ', chunk.root.head.pos_)

                    full_w_chunk = 1
                    for c_w in chunk:
                        if not c_w.is_alpha:  # or c_w.is_stop c_w.pos_ != "NOUN" or:
                            full_w_chunk = 0
                            break
                        # if '6' in c_w.text:
                        #     print()

                    if full_w_chunk:
                        if chunk.root.head.pos_ != 'VERB' and len(chunk) > 1:
                            head_w = chunk.text

                            if not head_w in nchunk_revs:
                                nchunk_revs[head_w] = set()
                            nchunk_revs[head_w].add(i)

                            val = nchunk_freq.get(head_w, 0)
                            nchunk_freq[head_w] = val + 1
                    # print(chunk.text, ' | ', chunk.root.text, ' | ', chunk.root.dep_, ' | ',
                    #       chunk.root.head.text, ' | ', chunk.root.head.pos_)
                # print('\n')

            i += 1

    pickle.dump((nchunk_freq, nchunk_revs, coref_freq, coref_revs, pos_freq, pos_revs, neg_freq, neg_revs),
                open('fr_nc-c-p-ne.p', "wb"))

    return nchunk_freq, nchunk_revs, coref_freq, coref_revs, pos_freq, pos_revs, neg_freq, neg_revs


if __name__ == '__main__':
    nchunk_freq, nchunk_revs, coref_freq, coref_revs, pos_freq, pos_revs, neg_freq, neg_revs = get_freqdicts()

    min_tresh = 4

    coref_freq_filt = filter_freqdicts(coref_freq, min_tresh * 7)
    coref_revs_filt_freq = filter_freqdicts(coref_revs, min_tresh * 7, setofvals=1)
    nchunk_revs_filt_freq = filter_freqdicts(nchunk_revs, min_tresh * 10, setofvals=1)
    nchunk_freq_filt = filter_freqdicts(nchunk_freq, min_tresh * 10)
    neg_revs_filt_freq = filter_freqdicts(neg_revs, min_tresh, setofvals=1)
    neg_freq_filt = filter_freqdicts(neg_freq, min_tresh)
    pos_revs_filt_freq = filter_freqdicts(pos_revs, min_tresh, setofvals=1)
    pos_freq_filt = filter_freqdicts(pos_freq, min_tresh)

    print('----- len pos: ' + str(len(pos_freq_filt)) + ' / ' + str(len(pos_revs_filt_freq)))
    print(list(pos_freq.items())[:5])
    print(list(pos_freq_filt.items())[:5])
    # print(list(pos_revs.items())[:5])
    print(list(pos_revs_filt_freq.items())[:5])
    # sys.exit(1)

    print('----- len neg: ' + str(len(neg_freq_filt)) + ' / ' + str(len(neg_revs_filt_freq)))
    print(list(neg_freq.items())[:5])
    print(list(neg_freq_filt.items())[:5])
    # print(list(neg_revs.items())[:5])
    print(list(neg_revs_filt_freq.items())[:5])

    print('----- len nchunk: ' + str(len(nchunk_freq_filt)) + ' / ' + str(len(nchunk_revs_filt_freq)))
    print(list(nchunk_freq.items())[:5])
    print(list(nchunk_freq_filt.items())[:5])
    # print(list(nchunk_revs.items())[:5])
    print(list(nchunk_revs_filt_freq.items())[:5])

    print('----- len coref: ' + str(len(coref_freq_filt)) + ' / ' + str(len(coref_revs_filt_freq)))
    print(list(coref_freq.items())[:5])
    print(list(coref_freq_filt.items())[:5])
    # print(list(coref_revs.items())[:5])
    print(list(coref_revs_filt_freq.items())[:5])

    showwordcloud(pos_revs_filt_freq)
    showwordcloud(neg_revs_filt_freq)
    # showwordcloud(nchunk_freq_filt)
    showwordcloud(nchunk_revs_filt_freq)
    # showwordcloud(coref_freq_filt)
    showwordcloud(coref_revs_filt_freq)

    # mat, doc_labels, vocab = pickle.load(open("mat_docs_topics.p", "rb"))
    # fig, ax = plt.subplots(figsize=(32, 8))
    # which_docs = random.sample(doc_labels, 30)
    # which_topics = list(set(vocab) & set(coref_freq_filt.keys()))     # random.sample(list(vocab), 30)
    # plot_doc_topic_heatmap(fig, ax, mat.toarray(), doc_labels, topic_labels=vocab,
    #                        which_documents=which_docs,
    #                        which_topics=which_topics)
    # plt.show()
