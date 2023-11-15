import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from globals import path_to_fixation_folder, path_to_text_with_tags_folder_bio, path_to_text_with_tags_folder_phy, reader_ids, biology_text_ids, \
    physics_text_ids
from utils import load_potec_words_sentences, load_fixation_per_reader

bio_words, bio_sentences = load_potec_words_sentences(path_to_text_with_tags_folder_bio)
phy_words, phy_sentences = load_potec_words_sentences(path_to_text_with_tags_folder_phy)

##-----------------------------For Biology-------------------------------##
fixation_scanpaths_bio = {tid: {} for tid in biology_text_ids}

for text_id_bio in biology_text_ids:
    for ppt_id_bio in reader_ids:
        suffix_bio = "reader" + ppt_id_bio + "_" + text_id_bio + "_fixations.txt"
        path_to_fixation_file_bio = os.path.join(path_to_fixation_folder, suffix_bio)

        reader_text_bio = load_fixation_per_reader(path_to_fixation_file_bio)

        scanpath_raw_bio = []

        for k_bio, (fix_ind_bio, char_ind_bio, fix_dur_bio) in enumerate(reader_text_bio):
            fixated_word_bio = ""

            for word_index_bio, info_bio in bio_words[text_id_bio].items():
                if char_ind_bio in info_bio["char_indices"]:
                    fixated_word_bio = info_bio["Word"]
                    word_index_in_text_bio = info_bio["WordIndexInText"]
                    sentence_index_bio = info_bio["SentenceIndex"]
                    word_index_in_sentence_bio = info_bio["WordIndexInSentence"]
                    technical_term_bio = info_bio["TechnicalTerm"]
                    SST_pos_bio = info_bio["SST_pos"]
                    punct_before_bio, punct_after_bio = info_bio["punct_before"], info_bio["punct_after"]
                    break
            
            scanpath_raw_bio.append((fixated_word_bio, fix_dur_bio, word_index_in_text_bio, sentence_index_bio,
                                 word_index_in_sentence_bio, technical_term_bio, SST_pos_bio, punct_before_bio, punct_after_bio))

        if ppt_id_bio == "0" and text_id_bio == "b0":  # get an example
            print("print var scanpath_raw")
        scanpath_postprocess_bio = []
        for fixation_triple_bio in scanpath_raw_bio:

            word_bio, fix_dur_bio, word_in_text_index_bio = fixation_triple_bio[0], fixation_triple_bio[1], fixation_triple_bio[2]
            sentence_index_bio, word_index_in_sentence_bio, technical_term_bio, SST_pos_bio = fixation_triple_bio[3], fixation_triple_bio[4], \
                fixation_triple_bio[5], fixation_triple_bio[6]
            punct_before_bio, punct_after_bio = fixation_triple_bio[7], fixation_triple_bio[8]
        
            if len(scanpath_postprocess_bio) == 0:
                scanpath_postprocess_bio.append(fixation_triple_bio)
                continue

            previous_word_bio, previous_index_bio = scanpath_postprocess_bio[-1][0], scanpath_postprocess_bio[-1][2]
            previous_dur_bio = scanpath_postprocess_bio[-1][1]
           
            if word_in_text_index_bio == previous_index_bio and len(scanpath_postprocess_bio) >= 1:
                scanpath_postprocess_bio[-1] = (previous_word_bio, fix_dur_bio+previous_dur_bio, word_in_text_index_bio, sentence_index_bio,
                                            word_index_in_sentence_bio, technical_term_bio, SST_pos_bio, punct_before_bio, punct_after_bio)
            else:
                scanpath_postprocess_bio.append(fixation_triple_bio)

        # put the created scanpath to the dictionary of scanpaths
        fixation_scanpaths_bio[text_id_bio][ppt_id_bio] = scanpath_postprocess_bio

# import pdb; pdb.set_trace()


##-----------------------------For Physics-------------------------------##

fixation_scanpaths_phy = {tid: {} for tid in physics_text_ids}

for text_id_phy in physics_text_ids:
    for ppt_id_phy in reader_ids:
        suffix_phy = "reader" + ppt_id_phy + "_" + text_id_phy + "_fixations.txt"
        path_to_fixation_file_phy = os.path.join(path_to_fixation_folder, suffix_phy)

        reader_text_phy = load_fixation_per_reader(path_to_fixation_file_phy)

        scanpath_raw_phy = []

        for k_phy, (fix_ind_phy, char_ind_phy, fix_dur_phy) in enumerate(reader_text_phy):
            fixated_word_phy = ""

            for word_index_phy, info_phy in phy_words[text_id_phy].items():
                if char_ind_phy in info_phy["char_indices"]:
                    fixated_word_phy = info_phy["Word"]
                    word_index_in_text_phy = info_phy["WordIndexInText"]
                    sentence_index_phy = info_phy["SentenceIndex"]
                    word_index_in_sentence_phy = info_phy["WordIndexInSentence"]
                    technical_term_phy = info_phy["TechnicalTerm"]
                    SST_pos_phy = info_phy["SST_pos"]
                    punct_before_phy, punct_after_phy = info_phy["punct_before"], info_phy["punct_after"]
                    break
            
            scanpath_raw_phy.append((fixated_word_phy, fix_dur_phy, word_index_in_text_phy, sentence_index_phy,
                                 word_index_in_sentence_phy, technical_term_phy, SST_pos_phy, punct_before_phy, punct_after_phy))

        if ppt_id_phy == "0" and text_id_phy == "p0":  # get an example
            print("print var scanpath_raw")
        scanpath_postprocess_phy = []
        for fixation_triple_phy in scanpath_raw_phy:

            word_phy, fix_dur_phy, word_in_text_index_phy = fixation_triple_phy[0], fixation_triple_phy[1], fixation_triple_phy[2]
            sentence_index_phy, word_index_in_sentence_phy, technical_term_phy, SST_pos_phy = fixation_triple_phy[3], fixation_triple_phy[4], \
                fixation_triple_phy[5], fixation_triple_phy[6]
            punct_before_phy, punct_after_phy = fixation_triple_phy[7], fixation_triple_phy[8]
        
            if len(scanpath_postprocess_phy) == 0:
                scanpath_postprocess_phy.append(fixation_triple_phy)
                continue

            previous_word_phy, previous_index_phy = scanpath_postprocess_phy[-1][0], scanpath_postprocess_phy[-1][2]
            previous_dur_phy = scanpath_postprocess_phy[-1][1]
           
            if word_in_text_index_phy == previous_index_phy and len(scanpath_postprocess_phy) >= 1:
                scanpath_postprocess_phy[-1] = (previous_word_phy, fix_dur_phy+previous_dur_phy, word_in_text_index_phy, sentence_index_phy,
                                            word_index_in_sentence_phy, technical_term_phy, SST_pos_phy, punct_before_phy, punct_after_phy)
            else:
                scanpath_postprocess_phy.append(fixation_triple_phy)

        # put the created scanpath to the dictionary of scanpaths
        fixation_scanpaths_phy[text_id_phy][ppt_id_phy] = scanpath_postprocess_phy
        

def plot_distribution_scanpath_lengths(list_bio_scanpaths, list_phys_scanpaths):
    domains = ["bio"] * len(list_bio_scanpaths) + ["phys"] * len(list_phys_scanpaths)
    scanpaths = list_bio_scanpaths + list_phys_scanpaths
    df = pd.DataFrame(zip(scanpaths, domains), columns=["Scanpath lengths", "Domain"])
    sns.set_theme()
    sns.histplot(df, x="Scanpath lengths", hue="Domain", kde=True)
    plt.show()

plot_distribution_scanpath_lengths(fixation_scanpaths_bio["b0"],fixation_scanpaths_phy["p0"])
