# -------------------- Cell --------------------
# Imports and setup
import collections
import random

import numpy as np
import pandas as pd

from qlynx.file_utils import store_pkl, load_pkl
from voynichlib.Corpus import Corpus
from voynichlib.ProbMassFunction import ProbMassFunction

# %reload_ext autoreload
# %autoreload 2


# -------------------- Cell --------------------
study_corpus_file = 'voynich_data/outputs/Study_Corpus_for_Positional_Tokens_Analysis.pkl'
corpus = Corpus.from_file(study_corpus_file)
transliteration = corpus.transliteration


# -------------------- Cell --------------------
# Define all Cohort Criteria 
criteria_by_c = collections.OrderedDict()
criteria_by_c['ALL'] = {}
criteria_by_c['MIDDLE'] =  {'token_positioning': 'middle', 
                            'last_token': False, 
                            'pre_drawing_token': False, 
                            'post_drawing_token': False, 
                            'paragraph_end_token': False, 
                            'paragraph_start_line': False}

criteria_by_c['TOP'] =     {'token_positioning': 'middle', 
                            'last_token': False, 
                            'pre_drawing_token': False, 
                            'post_drawing_token': False, 
                            'paragraph_end_token': False, 
                            'paragraph_start_line': True}

criteria_by_c['FIRST'] =   {'token_pos': 1,               
                            'last_token': False, 
                            'pre_drawing_token': False, 
                            'post_drawing_token': False, 
                            'paragraph_end_token': False, 
                            'paragraph_start_line': False}

criteria_by_c['LAST'] =    {'last_token': True,  
                            'pre_drawing_token': False, 
                            'post_drawing_token': False, 
                            'paragraph_end_token': False, 
                            'paragraph_start_line': False}

criteria_by_c['BEFORE'] =  {'last_token': False, 
                            'pre_drawing_token': True,  
                            'post_drawing_token': False, 
                            'paragraph_end_token': False, 
                            'paragraph_start_line': False}

criteria_by_c['AFTER'] =   {'last_token': False, 
                            'pre_drawing_token': False, 
                            'post_drawing_token': True,  
                            'paragraph_end_token': False, 
                            'paragraph_start_line': False}

criteria_by_c['SECOND'] =  {'token_pos': 2,               
                            'last_token': False, 
                            'pre_drawing_token': False, 
                            'post_drawing_token': False, 
                            'paragraph_end_token': False, 
                            'paragraph_start_line': False}

criteria_by_c['FOURTH'] =  {'token_pos': 4,               
                            'last_token': False, 
                            'pre_drawing_token': False, 
                            'post_drawing_token': False, 
                            'paragraph_end_token': False, 
                            'paragraph_start_line': False}


# -------------------- Cell --------------------
# Complile Dictionaries
all_cohorts = [k for k in criteria_by_c]
cohorts = [k for k in criteria_by_c]
cohorts.remove('ALL')

corpus_by_c = {}
pmfs_by_c = {}
pmfs_by_cw = {}
tokens_by_c = {}
tokens_by_cw = {}
token_ws_by_c = {}

glyph_pmfs_by_c = {}
glyphs_by_c = {}

for cohort, criteria in criteria_by_c.items():
    label = f"'{cohort}'"
    corpus_by_c[cohort] = Corpus.from_corpus(f'Scribe 1 - {cohort}', corpus, criteria=criteria, suppress_summary=True )
    df = corpus_by_c[cohort].tokens_df()
    pmfs_by_c[cohort] = ProbMassFunction(list(df['token']))
    tokens_by_c[cohort] = pmfs_by_c[cohort].values
    token_lengths = corpus_by_c[cohort].tokens_df()['token_length_min' ]
    print(f"Scribe 1, {label:20}: {np.mean(token_lengths):.2f} +/- {np.std(token_lengths):.2f}  [{np.min(token_lengths)}, {np.max(token_lengths)}]\t\t{len(token_lengths): >8,} obs")


    glyph_df = corpus_by_c[cohort].glyphs_df()
    glyph_pmfs_by_c[cohort] = ProbMassFunction(list(glyph_df['glyph']))
    glyphs_by_c[cohort] = glyph_pmfs_by_c[cohort].values
   
    token_ws_by_c[cohort]  = token_lengths

    pmfs_by_cw[cohort] = {}
    tokens_by_cw[cohort] = {}
    for w in range(1,11):
        tokens_for_pn = list(df[df['token_length_min'] == w]['token'])
        tokens_by_cw[cohort][w] = tokens_for_pn
        pmfs_by_cw[cohort][w] = ProbMassFunction(tokens_for_pn)
        pass
    pass
    
   
pass


# -------------------- Cell --------------------
print(f"Total tokens count in MIDDLE cohort = {pmfs_by_c['MIDDLE'].total_count:,}")
print(f"Number of occurrences of 'daiin' in MIDDLE cohort = {pmfs_by_c['MIDDLE'].count('daiin'):,}")
print(f"Probability of 'daiin' in MIDDLE cohort (no smoothing)= {pmfs_by_c['MIDDLE'].prob('daiin'):.2%}")
print(f"Probability of 'daiin' in MIDDLE cohort (laplace smoothing)= {pmfs_by_c['MIDDLE'].prob('daiin', smooth='laplace'):.2%}")
print(f"Probability of 'daiin' in MIDDLE cohort (minimal smoothing)= {pmfs_by_c['MIDDLE'].prob('daiin', smooth='minimal'):.2%}")
print()
print(f"Total tokens count in FIRST cohort = {pmfs_by_c['FIRST'].total_count:,}")
print(f"Number of occurrences of 'daiin' in FIRST cohort = {pmfs_by_c['FIRST'].count('daiin'):,}")
print(f"Probability of 'daiin' in FIRST cohort (no smoothing)= {pmfs_by_c['FIRST'].prob('daiin'):.2%}")
print(f"Probability of 'daiin' in FIRST cohort (laplace smoothing)= {pmfs_by_c['FIRST'].prob('daiin', smooth='laplace'):.2%}")
print(f"Probability of 'daiin' in FIRST cohort (minimal smoothing)= {pmfs_by_c['FIRST'].prob('daiin', smooth='minimal'):.2%}")





# -------------------- Cell --------------------
random.seed(20240115)


# -------------------- Cell --------------------
random_cohorts = ['RAND 1', 'RAND 2', 'RAND 3', 'RAND 4', 'RAND 5', 'RAND 6']
num_in_second = len(corpus_by_c['SECOND'].tokens_df())
num_in_pre = len(corpus_by_c['BEFORE'].tokens_df())
tokens_from_mid = corpus_by_c['MIDDLE'].tokens()
tokens_rand = {}
df = corpus_by_c['MIDDLE'].tokens_df()
for i, cohort in enumerate(random_cohorts):
    num_to_sample = num_in_second if i <3 else num_in_pre
    tokens_rand[cohort] = random.sample(tokens_from_mid, num_to_sample)
    pmfs_by_c[cohort] = ProbMassFunction(tokens_rand[cohort])
    pmfs_by_cw[cohort] = {}
    tokens_by_cw[cohort] = {}
    for w in range(1,11):
        tokens_by_cw[cohort][w] = [x for x in tokens_rand[cohort] if len(x)==w ]
        if not tokens_by_cw[cohort][w]:
            print(f"tokens_by_cw[{cohort}][{w}] is NONE")
        pmfs_by_cw[cohort][w] = ProbMassFunction(tokens_by_cw[cohort][w])
        pass
    pass
    token_lengths = []
    for token in tokens_rand[cohort]:
        token_length_df =df[df['token']==token]
        if len(token_length_df)==0:
            print(token)
        else:
            token_length = token_length_df['token_length_min'].iloc[0]
        token_lengths.append(token_length)
        pass
    token_ws_by_c[cohort] = token_lengths
pass
cohorts_with_randoms = cohorts + random_cohorts



# -------------------- Cell --------------------
def make_cohort_summary_table(cohorts):
    df = pd.DataFrame(columns=['Cohort', 'Folios', 'Lines', 'Tokens', 'Unique Tokens', 'Glyphs', 'Unique Glyphs'])#, index=['X', 'Y', 'Z'])
    for cohort in cohorts:
        if cohort.startswith('RAND'):
            corpus = corpus_by_c['MIDDLE'] 
            tokens = tokens_rand[cohort] 
            count_folios = '~'
            count_lines =  '~'
            count_tokens = len(tokens)
            count_glyphs = '~'
            ucount_tokens = len(set(list(tokens)))
            ucount_glyphs = '~'
            df.loc[len(df)] = [cohort,
                               count_folios,
                               count_lines,
                               count_tokens,
                               ucount_tokens,
                               count_glyphs,
                               ucount_glyphs]

        else:
            corpus = corpus_by_c[cohort] 
            count_folios = len(corpus.folios_df())
            count_lines = len(corpus.lines_df())
            count_tokens = len(corpus.tokens_df())
            count_glyphs = len(corpus.glyphs_df())
            ucount_tokens = len(set(list(corpus.tokens_df()['token'])))
            ucount_glyphs = len(list(set(corpus.glyphs_df()['glyph'])))
            df.loc[len(df)] = [cohort,
                               count_folios,
                               count_lines,
                               count_tokens,
                               ucount_tokens,
                               count_glyphs,
                               ucount_glyphs]
            
    
    df.to_csv('voynich_data/outputs/cohort_summary_data.csv')
    return df

make_cohort_summary_table(['ALL'] + cohorts_with_randoms)


# -------------------- Cell --------------------
df = corpus_by_c['ALL']
df.tokens_df()


# -------------------- Cell --------------------
df = corpus_by_c['FIRST']
df.tokens_df()


# -------------------- Cell --------------------
corpus_by_c['BEFORE'].tokens_df()


# -------------------- Cell --------------------
corpus_by_c['AFTER'].tokens_df()


# -------------------- Cell --------------------
token_cohort_data = {}
token_cohort_data['all_cohorts'] = all_cohorts
token_cohort_data['cohorts'] = cohorts
token_cohort_data['cohorts_with_randoms'] = cohorts_with_randoms

token_cohort_data['corpus_by_c'] = corpus_by_c
token_cohort_data['pmfs_by_c'] = pmfs_by_c
token_cohort_data['pmfs_by_cw'] = pmfs_by_cw
token_cohort_data['tokens_by_cw'] = tokens_by_cw
token_cohort_data['token_ws_by_c'] = token_ws_by_c

token_cohort_data['glyph_pmfs_by_c'] = glyph_pmfs_by_c
token_cohort_data['glyphs_by_c'] = glyphs_by_c

file_path = 'voynich_data/outputs/token_cohort_data.pkl'

store_pkl(token_cohort_data, file_path, ensure_dir = True)


# -------------------- Cell --------------------
token_cohort_data = None
x_token_cohort_data = load_pkl(file_path)
x_token_cohort_data.keys()


# -------------------- Cell --------------------


