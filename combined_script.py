# Z01.1 Preparation of Study Corpus.ipynb
print(80*"=")
print(80*"=")
print(f"||  Running Z01.1 Preparation of Study Corpus.ipynb")
print(80*"=")
print(80*"=")

# -------------------- Cell --------------------
# Imports and setup
from voynichlib.Corpus import Corpus
from voynichlib.Transliteration import Transliteration
import matplotlib.pyplot as plt
import seaborn as sns

# %reload_ext autoreload
# %autoreload 2


# -------------------- Cell --------------------
# Custom plot 
def plot_scribes_and_topics_heatmap(data, c, label_dict, filename:None):
    # Count the number of folios for each combination of scribe and illustration
    count_data = data.groupby(['fagin_davis_scribe', 'illustration']).size().reset_index(name='count')
    
    # Creating a pivot table with swapped rows and columns
    pivot_table = count_data.pivot_table(index='illustration', columns='fagin_davis_scribe', values='count', fill_value=0)

    # Rename the row labels using the provided dictionary
    pivot_table = pivot_table.rename(index=label_dict)

    # Mask for cells with 0 value
    mask = pivot_table == 0

    def annotate_heatmap(data, ax):
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                value = data.iloc[y, x]
                if value > 0:  # Annotate only if value is greater than 0
                    # Set font color to white if the value is over 90, otherwise black
                    font_color = 'white' if value > 90 else 'black'
                    plt.text(x + 0.5, y + 0.5, f'{int(value)}', 
                             horizontalalignment='center', 
                             verticalalignment='center', 
                             color=font_color,
                            fontsize=12)

    plt.figure(figsize=(4, 4))  # Adjusted figure size for better visibility of labels
    # Plotting the heatmap with the mask and line settings
    sns.heatmap(pivot_table, cmap=c, fmt="d", cbar=False, mask=mask, linecolor='black', linewidths=0.1)
    
    # Annotating the heatmap using the custom function
    annotate_heatmap(pivot_table, plt.gca())

    plt.xticks(rotation=0)  # Rotate column labels to horizontal
    plt.yticks(rotation=0)  # Rotate row labels to horizontal
    
    plt.xlabel("Scribe")
    plt.ylabel("Illustration Type")
    if filename:
        plt.savefig(filename,  bbox_inches='tight')

    plt.ion()  # Interactive mode on
    plt.show()
    plt.pause(2)
    plt.close()  
    return


# -------------------- Cell --------------------
transliteration_source_file = 'voynich_data/standard_ivtff/ZL_N_ext_Eva_3a.ivtff'


# -------------------- Cell --------------------
transliteration = Transliteration(transliteration_source_file)


# -------------------- Cell --------------------
corpus = Corpus(f"Entire Manuscript", transliteration, criteria = {})


# -------------------- Cell --------------------
whole_corpus_file = 'voynich_data/outputs/whole_corpus.pkl'
corpus.save(whole_corpus_file)


# -------------------- Cell --------------------
corpus_all = Corpus(f"VMS All", transliteration)
data = corpus_all.folios_df()[['folio','illustration','fagin_davis_scribe']]

label_dict = label_dict = {'A':'Astronomical',
              'B': 'Biological',
              'C': 'Cosmological',
              'H': 'Herbal',
              'P': 'Phamaceutical',
              'S': 'Stars (Recipes)',
              'T': 'Text Only',
              'Z': 'Zodiac'}

plot_scribes_and_topics_heatmap(data, 'Greens', label_dict, 'voynich_data/outputs/F_Folios_by_Scribe_and_Illustration_Type.png')


# -------------------- Cell --------------------
corpus = Corpus(f"Study Corpus for Positional Tokens Analysis", transliteration, criteria = {
    'fagin_davis_scribes' : ['1'],
    'illustrations':'H',
    'unambiguous_token': True,
    'paragraph_end_token': False,
    'locus_generic_types': 'P'})


# -------------------- Cell --------------------
study_corpus_file = 'voynich_data/outputs/Study_Corpus_for_Positional_Tokens_Analysis.pkl'


# -------------------- Cell --------------------
corpus.save(study_corpus_file)


# -------------------- Cell --------------------
corpus.folios()


# -------------------- Cell --------------------
corpus_x = Corpus.from_file(study_corpus_file)


# -------------------- Cell --------------------




# Z01.2 Token Cohorts.ipynb
print(80*"=")
print(80*"=")
print(f"||  Running Z01.2 Token Cohorts.ipynb")
print(80*"=")
print(80*"=")

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




# Z01.3 Token Length Analysis.ipynb
print(80*"=")
print(80*"=")
print(f"||  Running Z01.3 Token Length Analysis.ipynb")
print(80*"=")
print(80*"=")

# -------------------- Cell --------------------
# Imports and setup
import pandas as pd
import scipy.stats as stats

from qlynx.file_utils import load_pkl
from qlynx.plot_utils import plot_combined_curves
from qlynx.plot_utils import plot_heatmap_Z001
from qlynx.plot_utils import plot_adjacent_histograms_with_binomial_curves
from qlynx.stats_utils import *
from voynichlib.ProbMassFunction import ProbMassFunction

# %reload_ext autoreload
# %autoreload 2


# -------------------- Cell --------------------
bayes_factor_binomial(0, 300, 0, .01)


# -------------------- Cell --------------------
file_path = 'voynich_data/outputs/token_cohort_data.pkl'
token_cohort_data = load_pkl(file_path)

all_cohorts = token_cohort_data['all_cohorts']
cohorts = token_cohort_data['cohorts']
cohorts_with_randoms = token_cohort_data['cohorts_with_randoms']

corpus_by_c = token_cohort_data['corpus_by_c']
pmfs_by_c = token_cohort_data['pmfs_by_c']
pmfs_by_cw = token_cohort_data['pmfs_by_cw']
tokens_by_cw = token_cohort_data['tokens_by_cw']
token_ws_by_c = token_cohort_data['token_ws_by_c']

glyph_pmfs_by_c = token_cohort_data['glyph_pmfs_by_c']
glyphs_by_c = token_cohort_data['glyphs_by_c']


# -------------------- Cell --------------------


# -------------------- Cell --------------------
def make_token_length_summary_table(cohorts):
    df = pd.DataFrame(columns=['Cohort', 'Mean', 'StDev', 'Min', 'Max', 'Observations'])
    for cohort in cohorts:   
        # if cohort.startswith('Rand'):
            
        token_lengths = token_ws_by_c[cohort]
        # corpus_by_c[cohort].tokens_df()['token_length_min' ]
        df.loc[len(df)] = [cohort,
                           np.mean(token_lengths),
                           np.std(token_lengths),
                           np.min(token_lengths),
                           np.max(token_lengths),
                           len(token_lengths)]
        pass    
    df.to_csv('voynich_data/outputs/token_length_summary_data.csv')
    return df
df = make_token_length_summary_table(['ALL'] + cohorts_with_randoms)
df


# -------------------- Cell --------------------
def make_token_length_summary_table(cohorts):
    df = pd.DataFrame(columns=['Cohort', 'Mean', 'Std', 'Min', 'Max', 'Num_Obs'])
    for cohort in all_cohorts:   
        # if cohort.startswith('Rand'):
            
        token_lengths = token_ws_by_c[cohort]
        # corpus_by_c[cohort].tokens_df()['token_length_min' ]
        df.loc[len(df)] = [cohort,
                           np.mean(token_lengths),
                           np.std(token_lengths),
                           np.min(token_lengths),
                           np.max(token_lengths),
                           len(token_lengths)]
        pass    
    df.to_csv('voynich_data/outputs/token_length_summary_data.csv')
    return df
df = make_token_length_summary_table(['ALL'] + cohorts_with_randoms)
df


# -------------------- Cell --------------------
def do_cohort_similarity_analysis(test_type: str,
                                fit_type: str,
                                significance_threshold: float = 0.01,
                                additional_spec:dict = {}, 
                                cutoff:int = 20,
                                smooth:str = None,
                                pmf_filename: str = None,
                                matrix_filename: str = None):
    num_cohorts = len(cohorts)
    p_matrix = np.zeros((num_cohorts, num_cohorts))

    for i  in range(num_cohorts):
        ref_tokens_ws = token_ws_by_c[cohorts[i]]
        for j in range(i+1):
            tokens_ws = token_ws_by_c[cohorts[j]]
            if test_type == 'ks':
                stat, pvalue = kolmogorov_smirnov_test(ref_tokens_ws,tokens_ws)  
                signficance_stat = pvalue
            elif test_type == 'wt':
                tresult = stats.ttest_ind(a=ref_tokens_ws, b=tokens_ws, equal_var=False)
                if  np.isnan(tresult.pvalue):
                    continue
                signficance_stat = tresult.pvalue
                pass
            elif test_type == 'chi2':
                chi2, p, dof, expected = chi_square_test_from_observations(ref_tokens_ws, tokens_ws, min_bin_size=5)                
                signficance_stat = p
            elif test_type == 'bayes':
                # bayes_factor = find_bayes_factor(tokens_ws, ref_tokens_ws)
                pmf1 = ProbMassFunction(list(tokens_ws)) 
                pmf2 = ProbMassFunction(list(ref_tokens_ws))
                vals = pmf1.values
                probs1 = {k: pmf1.prob(k, smooth='minimal') for k,v in pmf2.pmf.items()}
                probs2 = {k: pmf2.prob(k, smooth='minimal') for k,v in pmf2.pmf.items()}
                bf = calculate_bayes_factor(probs1, probs2)
                signficance_stat=bf
            else:
                raise Exception("Invalid test type")
            
            p_matrix[i,j] = p_matrix[j,i] = signficance_stat
            pass
    token_ws_by_c_no_random = {k:v for k,v in token_ws_by_c.items() if not k.startswith('RAND')}
    

    plot_combined_curves(token_ws_by_c_no_random, 
                         fit_model=fit_type, 
                         title=f"Probability Mass Distribution of Token Glyph-Counts (Binomial Fits)", 
                         cutoff=cutoff,
                         filename=pmf_filename,
                         autoclose=True,
                        )

    includes = [len(v) > cutoff for k,v in token_ws_by_c.items() if k in cohorts]
    # p_matrix_binary_x =  p_matrix_binary[includes][:, includes]
    p_matrix_x =  p_matrix[includes][:, includes]
    cohorts_x =  [cohorts[i] for i in range(len(cohorts)) if includes[i]]

    if len(cohorts_x) < 2:
        print(f"No Data for p_matrix plots.")
        return

    plot_heatmap_Z001(p_matrix_x, 
                        cohorts_x, 
                        threshold=0.01,
                      title='',
                        # title = f"Chi-Squared Statistical Significance Matrix",
                 lo_color = 'pink', hi_color = 'green', annot_max = 12, filename=matrix_filename, autoclose=True)
    return 

       


# -------------------- Cell --------------------
do_cohort_similarity_analysis('chi2', 'binomial', 
                              significance_threshold=.01,                               
                              cutoff=50,
                              smooth='minimal',
                              pmf_filename = 'voynich_data/outputs/F_PMFs_glyph_counts_ALL.png',
                             matrix_filename = 'voynich_data/outputs/F_Significance_Matrix.png')


# -------------------- Cell --------------------
plots_to_do = [('ALL', 'MIDDLE'), ( 'MIDDLE', 'TOP'), ('MIDDLE', 'FIRST'), ('MIDDLE', 'BEFORE'),('MIDDLE', 'SECOND'), ('MIDDLE', 'AFTER')]
for i in range(len(all_cohorts)):
    x2 = all_cohorts[i]
    d2 = token_ws_by_c[x2]
    for j in range(i):
        x1 = all_cohorts[j]
        if not (x1,x2) in plots_to_do:
            continue
        d1 = token_ws_by_c[x1]
        plot_adjacent_histograms_with_binomial_curves(d1, d2, label1=x1, label2=x2, title='', #f"Comparison of {x1} and {x2}",
                                                     xlabel="Token Length in terms of Glyph Count",
                                                     ylabel="Probabilty Mass",
                                                     filename=f"voynich_data/outputs/F_PMF_Compare_{x1}_{x2}.png",
                                                      autoclose=True)
        


# -------------------- Cell --------------------




# Z01.4 Token Propensities By Location.ipynb
print(80*"=")
print(80*"=")
print(f"||  Running Z01.4 Token Propensities By Location.ipynb")
print(80*"=")
print(80*"=")

# -------------------- Cell --------------------
# Imports and setup
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, HTML
import os

from qlynx.file_utils import load_pkl
from qlynx.stats_utils import *
from qlynx.display_utils import render_html_to_image
from voynichlib.utils import display_voynichese

# %reload_ext autoreload
# %autoreload 2
 


# -------------------- Cell --------------------
do_parametric_studies = True
MAX_BAYES = np.exp(10)
MAX_PROPENSITY = 999
THRESHOLDS = {
    'p_value': 0.01,
    'ln_bayes_factor': 5
}
THRESHOLDS['bayes_factor'] = np.exp(THRESHOLDS['ln_bayes_factor'])
reference_cohort = 'MIDDLE'
# smooth = 'laplace'
# smooth = 'laplace'
smooth = None



# -------------------- Cell --------------------
file_path = 'voynich_data/outputs/token_cohort_data.pkl'
token_cohort_data = load_pkl(file_path)

cohorts = token_cohort_data['cohorts']
cohorts_with_randoms = token_cohort_data['cohorts_with_randoms']

corpus_by_c = token_cohort_data['corpus_by_c']
pmfs_by_c = token_cohort_data['pmfs_by_c']
token_ws_by_c = token_cohort_data['token_ws_by_c']

glyph_pmfs_by_c = token_cohort_data['glyph_pmfs_by_c']
glyphs_by_c = token_cohort_data['glyphs_by_c']


# -------------------- Cell --------------------
def get_top_vocabulary_tokens_lengths_dict(cohort, N_v:int=None):
    tokens = pmfs_by_c[cohort].values
    if not N_v:
        tokens = tokens[:N_v]
    token_lengths_dict = {}
    df = corpus_by_c[cohort].tokens_df()
    for token in tokens:
        df_token = df[df['token'] == token]
        token_length = df_token['token_length_min'].iloc[0]
        token_lengths_dict[token] = token_length
        pass
    pass
    return token_lengths_dict


# -------------------- Cell --------------------
def compile_token_propensity_df(target_cohort, reference_cohort, p_value_threshold, bayes_threshold):
    top_token_length_dict = get_top_vocabulary_tokens_lengths_dict(reference_cohort)
    df = pd.DataFrame(columns = ['token', 'glyph_count', 'N_ref', 'n_ref', 'N_x', 'n_x', 'p_ref', 'p_x', 'p_value', 'sig_p_value', 'sig_BF', 'propensity', 'bayes', 'binom_stat_le', 'binom_stat_gt'])
    for token, w in top_token_length_dict.items():
        pmf_ref = pmfs_by_c[reference_cohort]
        N_ref = pmf_ref.total_count
        n_ref = pmf_ref.count(token) if N_ref > 0 else 0
        p_ref = pmf_ref.prob(token, smooth=smooth)

        pmf_x = pmfs_by_c[target_cohort]                        
        N_x = pmf_x.total_count
        n_x = pmf_x.count(token) if N_x > 0 else 0
        p_x = pmf_x.prob(token, smooth=smooth)

        p_value = calculate_binomial_probability(n_x, N_x, p_ref)

        bayes_factor = bayes_factor_binomial(n_x, N_x, p_x, p_ref)
        bayes_factor = min(MAX_BAYES, bayes_factor)

        binom_stat_le =  binom.cdf(n_x, N_x, p_ref)        
        binom_stat_gt =  binom.cdf(n_x, N_x, 1. -p_ref)        

        if target_cohort.startswith('Rand'):
            propensity = 1.
        else: 
            propensity = p_x/p_ref  if p_ref > 0 else MAX_PROPENSITY
            pass
        verdict_p_value = p_value < p_value_threshold
        verdict_bayes_factor = bayes_factor > bayes_threshold
        df.loc[len(df)] = [token,
                           top_token_length_dict[token],
                           N_ref,
                           n_ref,
                           N_x,
                           n_x,
                           p_ref,
                           p_x,
                           p_value,
                           verdict_p_value,
                           verdict_bayes_factor,
                           np.round(propensity,1),
                           bayes_factor,
                          binom_stat_le,
                          binom_stat_gt]
        pass
    df.set_index('token', inplace=True)
    pass
    return df



# -------------------- Cell --------------------
styles = {
    'ALL':['grey','-',1],
    'MIDDLE':['red','-',1],
    'TOP':['red','-.',1],
    'FIRST':['green','-',2],
    'SECOND':['orange','-.',2],
    'THIRD':['orange','-.',2],
    'FOURTH':['orange','-.',2],
    'BEFORE':['blue','--',2],
    'AFTER':['blue','--',1],
    'LAST':['green','-',1],
    'RAND 1':['grey',':',1],
    'RAND 2':['grey',':',1], 
    'RAND 3':['grey',':',1],
    'RAND 4':['grey',':',1],
    'RAND 5':['grey',':',1],
    'RAND 6':['grey',':',1],
}        

p_thresholds_x = [10,9,8,7,6,5,4,3,2]
p_thresholds1 = [.01*x for x in p_thresholds_x]
p_thresholds2 = [.001*x for x in p_thresholds_x]
p_thresholds3 = [.0001*x for x in p_thresholds_x]
p_thresholds4 = [.00001*x for x in p_thresholds_x]
p_thresholds = p_thresholds1 + p_thresholds2 + p_thresholds3 + p_thresholds4

bayes_thresholds_x = [1,2,3,4,5,6,7,8,9]
bayes_thresholds1 = [1*x for x in bayes_thresholds_x]
bayes_thresholds2 = [10*x for x in bayes_thresholds_x]
bayes_thresholds3 = [100*x for x in bayes_thresholds_x]
bayes_thresholds4 = [1000*x for x in bayes_thresholds_x]
bayes_thresholds = bayes_thresholds1 + bayes_thresholds2 + bayes_thresholds3 + bayes_thresholds4


# -------------------- Cell --------------------
def plot_threshold_parametric_data(stat_type, tokens_as_list, filename:str = None):   
    for cohort in cohorts_with_randoms:
        if cohort=='ALL':
            continue
        if cohort=='MIDDLE':
            continue
        if stat_type == 'p_value':
            plt.plot(p_thresholds, 
                 tokens_as_list[cohort].values(), 
                 label=cohort, 
                 color=styles[cohort][0], 
                 linestyle=styles[cohort][1], 
                 linewidth=styles[cohort][2])
        elif stat_type == 'bayes':
            plt.plot(bayes_thresholds, 
                     tokens_as_list[cohort].values(), 
                     label=cohort, 
                     color=styles[cohort][0], 
                     linestyle=styles[cohort][1], 
                     linewidth=styles[cohort][2])
        pass
    plt.legend()
    if stat_type == 'p_value':
        plt.xlabel('p-value Threshold')
        plt.ylim((0,50))
        plt.xlim((0,0.1))
        threshold = THRESHOLDS['p_value']
    elif stat_type == 'bayes':
        plt.xlabel('B Threshold')
        plt.xscale('log')
        plt.ylim((0,50))
        plt.xlim((3,10000))
        threshold = THRESHOLDS['bayes_factor']
    pass
    plt.axvline(x=threshold, color='r', linestyle='--')
    plt.ylabel('Count of Tokens with Significant Propensity')
    if filename:
        plt.savefig(filename,  bbox_inches='tight')

    plt.ion()  # Interactive mode on
    plt.show()
    plt.pause(2)
    plt.close()  


# -------------------- Cell --------------------
def compile_threshold_parametic_data(stat_type, use_both_thresholds:bool=False):
    print(f"Parametric Study of {stat_type}")
    num_propensity_tokens_by_cohort_and_threshold = {}
    for cohort in cohorts_with_randoms:
        if cohort=='ALL':
            continue
        print(f"\tCompiling {cohort}")
        num_propensity_tokens_by_cohort_and_threshold[cohort] = {}
        if stat_type == 'p_value':
            for _p_value_threshold in p_thresholds:
                df = compile_token_propensity_df(cohort, reference_cohort, _p_value_threshold, THRESHOLDS['bayes_factor'])
                if use_both_thresholds:
                    token_propensity_df = df[(df['sig_p_value']) & (df['sig_BF'])]
                else:
                    token_propensity_df = df[(df['sig_p_value'])]
                num_propensity_tokens_by_cohort_and_threshold[cohort][_p_value_threshold] = len(token_propensity_df)
                pass
        elif stat_type == 'bayes':
            for _bayes_threshold in bayes_thresholds:
                df = compile_token_propensity_df(cohort, reference_cohort, THRESHOLDS['p_value'], _bayes_threshold)
                if use_both_thresholds:
                    token_propensity_df = df[(df['sig_p_value']) & (df['sig_BF'])]
                else:
                    token_propensity_df = df[(df['sig_BF'])]
                num_propensity_tokens_by_cohort_and_threshold[cohort][_bayes_threshold] = len(token_propensity_df)
                pass
            pass
            
    pass
    return num_propensity_tokens_by_cohort_and_threshold


# -------------------- Cell --------------------
if do_parametric_studies:
    stat_type = 'p_value'
    num_propensity_tokens_p_value = compile_threshold_parametic_data(stat_type, use_both_thresholds=False)
    print("This plot uses both p_value and Bayes Factor.")
    plot_threshold_parametric_data(stat_type, 
                                   num_propensity_tokens_p_value)


# -------------------- Cell --------------------
if do_parametric_studies:
    stat_type = 'p_value'
    num_propensity_tokens_p_value = compile_threshold_parametic_data(stat_type) 
    print("This plot uses the p_value threshold only.")
    plot_threshold_parametric_data(stat_type, 
                                   num_propensity_tokens_p_value,  
                                   filename = 'voynich_data/outputs/F_Threshold_Parametric_p_values.png')


# -------------------- Cell --------------------
if do_parametric_studies:
    stat_type = 'bayes'
    num_propensity_tokens_bayes = compile_threshold_parametic_data(stat_type) 
    print("This plot uses the Bayes Factor threshold only.")
    plot_threshold_parametric_data(stat_type, 
                                   num_propensity_tokens_bayes, 
                                   filename = 'voynich_data/outputs/F_Threshold_Parametric_bayes.png')


# -------------------- Cell --------------------
token_propensity_dfs = {}
print(f"smoothing: {smooth}")
print(f"Summary Dataframes,   p_value_threshold= {THRESHOLDS['p_value']} , bayes_threshold={THRESHOLDS['bayes_factor']:.1f}")
N_tokens_df = pd.DataFrame(columns = ['cohort',  'N_p', 'N_p_af', 'N_p_av', 'N_b', 'N_b_af', 'N_b_av',  'N_either', 'N_both'])

for cohort in cohorts_with_randoms:
    if cohort == 'MIDDLE':
        continue
    token_propensity_dfs[cohort] = compile_token_propensity_df(cohort, 
                                                               reference_cohort, 
                                                               THRESHOLDS['p_value'], 
                                                               THRESHOLDS['bayes_factor'])
    df = token_propensity_dfs[cohort]
    N_p = len(df[ df['sig_p_value']])
    N_p_af =  len(df[ df['sig_p_value'] & (df['propensity']>0) ])
    N_p_av =  len(df[ df['sig_p_value'] & (df['propensity']<0) ])
    N_b = len(df[ df['sig_BF']])
    N_b_af =  len(df[ df['sig_BF'] & (df['propensity']>0) ])
    N_b_av =  len(df[ df['sig_BF'] & (df['propensity']<0) ])
    N_either =  len(df[ df['sig_BF'] | df['sig_p_value'] ])
    N_both =  len(df[ df['sig_BF'] & df['sig_p_value'] ])
    N_tokens_df.loc[len(N_tokens_df)] = [cohort,
                       N_p,
                       N_p_af,
                       N_p_av,
                       N_b,
                       N_b_af,
                       N_b_av,
                       N_either,
                       N_both]
                                             
    pass
    # print(80*'-')
pass
N_tokens_df


# -------------------- Cell --------------------
def extract_df(cohort, stat_type, component_type):
    print(f"'{cohort}' Compared to 'MIDDLE'")
    if component_type == 'tokens':
        df = token_propensity_dfs[cohort]
    elif component_type == 'glyphs':
       df = glyph_propensity_dfs[cohort]

    if stat_type == 'p_value':
        df = df[df['sig_p_value']]
        df.sort_values(by='p_value', ascending=False)
    elif stat_type == 'bayes':
        df = df[df['sig_BF']]        
        df.sort_values(by='bayes', ascending=False)
    elif stat_type == 'both':
        df = df[(df['sig_p_value']) & (df['sig_BF'])]
        
    print(len(df))
    return df



# -------------------- Cell --------------------
extract_df('SECOND', 'both', 'tokens')


# -------------------- Cell --------------------
extract_df('TOP', 'both', 'tokens')


# -------------------- Cell --------------------
extract_df('FIRST', 'both', 'tokens')


# -------------------- Cell --------------------
extract_df('LAST', 'both', 'tokens')


# -------------------- Cell --------------------
extract_df('BEFORE', 'both', 'tokens')


# -------------------- Cell --------------------
extract_df('AFTER', 'both', 'tokens')


# -------------------- Cell --------------------
extract_df('FOURTH', 'p_value', 'tokens')


# -------------------- Cell --------------------
cohort_title_dict = {
'ALL':'All in Corpus',
'MIDDLE':'Middle Positions',
'TOP':'Top Lines of Paragraphs',
'FIRST': 'First Position on a Line',
'SECOND': 'Second Position on a Line',
'THIRD': 'Third Position on a Line',
'FOURTH': 'Fourth Position on a Line',
'BEFORE': 'Immediately Before a Drawing',
'AFTER':'Immediately After a Drawing',
'LAST': 'Last Position on a Line',
'RAND 1':'Random Tokens Cohort',
'RAND 2':'Random Tokens Cohort', 
'RAND 3':'Random Tokens Cohort',
'RAND 4':'Random Tokens Cohort',
'RAND 5':'Random Tokens Cohort',
'RAND 6':'Random Tokens Cohort',
}    

def filter_and_sort_dataframe(df, propensity_col, sig_p_value_col, sig_BF_col):
    # Filter rows where at least one of sig_p_value or sig_BF is True
    filtered_df = df[(df[sig_p_value_col]) & (df[sig_BF_col])]
    # filtered_df = df[(df[sig_p_value_col])]

    # Splitting the DataFrame based on propensity values
    df_greater_than_zero = filtered_df[filtered_df[propensity_col] > 1].sort_values(by=propensity_col, ascending=False)
    df_less_than_zero = filtered_df[filtered_df[propensity_col] < 1].sort_values(by=propensity_col, ascending=True)

    # Concatenating the two DataFrames
    result_df = pd.concat([df_greater_than_zero, df_less_than_zero])

    return result_df

def display_cohort_tendency_summary(cohort: str, component:str, stat_type:str, file_name: str = None, width:int=None, height:int=None):
    if component == 'tokens':
        df = token_propensity_dfs[cohort].sort_values(by='propensity', ascending=False)
        table_title = f"Positional Tendency Tokens<br>{cohort_title_dict[cohort]}"
        num_tokens_in_target = pmfs_by_c[cohort].total_count
        num_tokens_in_ref = pmfs_by_c[reference_cohort].total_count
        
        # num_tokens_in_target = len(corpus_by_c[cohort].tokens_df())
        # num_tokens_in_ref = len(corpus_by_c[reference_cohort].tokens_df())
        component_text = 'Tokens'
        
    elif component == 'glyphs':
        df = glyph_propensity_dfs[cohort].sort_values(by='propensity', ascending=False)
        table_title = f"Positional Tendency Glyphs<br>{cohort_title_dict[cohort]}"
        num_tokens_in_target = len(corpus_by_c[cohort].glyphs_df())
        num_tokens_in_ref = len(corpus_by_c[reference_cohort].glyphs_df())
        component_text = 'Glyphs'
        pass
    pass

    df = filter_and_sort_dataframe(df,'propensity', 'sig_p_value', 'sig_BF')
    # display_stat_type = 'BOTH'
        
    
    html_top = """
<html>
<head>
    <style>
        h3 {
            margin-left: auto;
            margin-right: auto;
        }
        table {
            border: 3px solid black;
            border-collapse: collapse;
            margin-left: auto;
            margin-right: auto;
        }

        th, td {
            border: 1px solid black;
            text-align: center;
        }

       .header-row {
            background-color: #7AA4F8;
        }     
        table td, table th {
            padding-left: 5px;
            padding-right: 5px;
        }
        
        tbody tr:nth-child(even) {
            background-color: #FEEFC2; /*#FFFFD9; light beige for odd rows */
        }

        tbody tr:nth-child(odd) {
            background-color: white; /* white for even rows */
        }
    </style>
</head>
<body>"""
    html_bottom = """
</body>
</html>"""
    html = ''
    # # Start the HTML table
    # html += f"<h2>{table_title}</h2>\n"
    # html += f"Reference Cohort: {reference_cohort}<br>\n"
    # html += f"Total Count in Reference Cohort: {num_tokens_in_ref}<br>\n"
    # html += f"Total Count in Target Cohort: {num_tokens_in_target}<br>\n"
    # html += f"Total Count Selected: {len(df)}\n"
    html_table_top = """
<table style='width:600px'>
    <tr>
        <th class='header-row' colspan=1 rowspan=2 style='text-align: center;'>Tilt</th>
        <th class='header-row' colspan=2 style='text-align: center;'>Token</th>
        <th class='header-row' colspan=2 style='text-align: center;'>Counts</th>
        <th class='header-row' colspan=3 style='text-align: center;'>Stats</th>
    </tr>
    <tr>
        <th class='header-row' >Voynichese</th>
        <th class='header-row' >Eva-</th>
        <th class='header-row' >expected</th>
        <th class='header-row' >observed</th>
        <th class='header-row' >Propensity</th>
        <th class='header-row' ><i>p</i>-value</th>
        <th class='header-row' ><i>log(B)</i></th>
    </tr>"""    
    html_1 = html_table_top
    # html_2 = html_table_top
    num_affinitive = len(df[df['propensity'] >= 1])
    num_aversive = len(df[df['propensity'] < 1])
    
    color = 'black' 
    # Fill the table rows
    i=-1
    for index, row in df.iterrows(): 
        i += 1
        propensity = row['propensity']
        # ln_propensity = np.log(propensity)
        # if propensity < 1:
        #     continue
        voynichese_value = display_voynichese(text=index, render=False)

        prob_ref= row['p_ref']
        p_value = row['p_value']
        prob_x = row['p_x']
        N_x = row['N_x']
        
        observed_count = int(row['n_x'])
        expected_count =  int(np.round(prob_ref * num_tokens_in_target))
        # expected_count =  prob_ref * num_tokens_in_target
        # propensity = f"{propensity:.1f}" if observed_count > 0 else '&infin;'
        bayes = row['bayes']
        bayes = f"{np.log(bayes):.1f}" if bayes > 0 else '0'
        if bayes=='10.0':
            bayes = '>10'
            

        starp = starb = ''
        if not row['sig_p_value']:
            starp = '*'
        if i == 0:
            color = 'green'
            html_1 += f"""
<tr>
    <td rowspan={num_affinitive} style="background-color:white;color:{color};"><b>Affinitive</b></td>"""
            pass            
        elif i == num_affinitive:
            color = 'red'
            html_1 += f"""
<tr style="border-top: 3px solid black;">
    <td rowspan={num_aversive} style="background-color:white;color:{color};"><b>Aversive</b></td>"""
        else:
            html_1 += f"""
<tr>"""
            pass
        
        html_1 += f"""            
    <td style='color:{color};'>{voynichese_value}</td>
    <td>{index}</td>
    <td>{expected_count}</td>
    <td>{observed_count}</td>
    <td>{propensity:.1f}</td>
    <td>{p_value:.6f}{starp}</td>
    <td>{bayes}{starb}</td>
</tr>"""
            

    html_1 += """
</table>"""
    html += html_1


    # Display the HTML table
    display(HTML(html))
    if file_name:
        current_dir = os.getcwd()
        print(f"current_dir = {current_dir}")
        html_filename = file_name + '.html'
        absolute_html_file_path = os.path.join(current_dir, html_filename)
        png_filename = file_name + '.png'
        absolute_png_file_path = os.path.join(current_dir, png_filename)
        with open(html_filename, 'w') as file:
            file.write(html_top + html + html_bottom)
            print(f"Wrote {html_filename}")
        print(f"absolute_html_file_path = {absolute_html_file_path}")
        render_html_to_image(absolute_html_file_path, absolute_png_file_path, width=width, height=height+129, crop=True)



# -------------------- Cell --------------------
display_cohort_tendency_summary('TOP', 'tokens', 'bayes', 'voynich_data/outputs/T_token_propensities_TOP_bayes', width=630, height=500)
display_cohort_tendency_summary('FIRST', 'tokens', 'bayes', 'voynich_data/outputs/T_token_propensities_FIRST_bayes', width=630, height=1000)
display_cohort_tendency_summary('LAST', 'tokens', 'bayes', 'voynich_data/outputs/T_token_propensities_LAST_bayes', width=630, height=1000)
display_cohort_tendency_summary('BEFORE', 'tokens', 'bayes', 'voynich_data/outputs/T_token_propensities_BEFORE_bayes', width=630, height=500)
display_cohort_tendency_summary('AFTER', 'tokens', 'bayes', 'voynich_data/outputs/T_token_propensities_AFTER_bayes', width=630, height=500)
display_cohort_tendency_summary('SECOND', 'tokens', 'bayes', 'voynich_data/outputs/T_token_propensities_SECOND_bayes', width=630, height=500)
display_cohort_tendency_summary('FOURTH', 'tokens', 'bayes', 'voynich_data/outputs/T_token_propensities_FOURTH_bayes', width=630, height=500)
display_cohort_tendency_summary('RAND 1', 'tokens', 'bayes', 'voynich_data/outputs/T_token_propensities_RAND1_bayes', width=630, height=500)
display_cohort_tendency_summary('RAND 2', 'tokens', 'bayes', 'voynich_data/outputs/T_token_propensities_RAND2_bayes', width=630, height=500)
display_cohort_tendency_summary('RAND 3', 'tokens', 'bayes', 'voynich_data/outputs/T_token_propensities_RAND3_bayes', width=630, height=500)
display_cohort_tendency_summary('RAND 4', 'tokens', 'bayes', 'voynich_data/outputs/T_token_propensities_RAND4_bayes', width=630, height=500)
display_cohort_tendency_summary('RAND 5', 'tokens', 'bayes', 'voynich_data/outputs/T_token_propensities_RAND5_bayes', width=630, height=500)
display_cohort_tendency_summary('RAND 6', 'tokens', 'bayes', 'voynich_data/outputs/T_token_propensities_RAND6_bayes', width=630, height=500)


# -------------------- Cell --------------------
display_cohort_tendency_summary('TOP', 'tokens', 'p_value', 'voynich_data/outputs/T_token_propensities_TOP', width=630, height=500)
display_cohort_tendency_summary('FIRST', 'tokens', 'p_value', 'voynich_data/outputs/T_token_propensities_FIRST', width=630, height=1600)
display_cohort_tendency_summary('LAST', 'tokens', 'p_value', 'voynich_data/outputs/T_token_propensities_LAST', width=630, height=1000)
display_cohort_tendency_summary('BEFORE', 'tokens', 'p_value', 'voynich_data/outputs/T_token_propensities_BEFORE', width=630, height=500)
display_cohort_tendency_summary('AFTER', 'tokens', 'p_value', 'voynich_data/outputs/T_token_propensities_AFTER', width=630, height=500)
display_cohort_tendency_summary('SECOND', 'tokens', 'p_value', 'voynich_data/outputs/T_token_propensities_SECOND', width=630, height=500)
display_cohort_tendency_summary('FOURTH', 'tokens', 'p_value', 'voynich_data/outputs/T_token_propensities_FOURTH', width=630, height=500)
display_cohort_tendency_summary('RAND 1', 'tokens', 'p_value', 'voynich_data/outputs/T_token_propensities_RAND1', width=630, height=500)
display_cohort_tendency_summary('RAND 2', 'tokens', 'p_value', 'voynich_data/outputs/T_token_propensities_RAND2', width=630, height=500)
display_cohort_tendency_summary('RAND 3', 'tokens', 'p_value', 'voynich_data/outputs/T_token_propensities_RAND3', width=630, height=500)
display_cohort_tendency_summary('RAND 4', 'tokens', 'p_value', 'voynich_data/outputs/T_token_propensities_RAND4', width=630, height=500)
display_cohort_tendency_summary('RAND 5', 'tokens', 'p_value', 'voynich_data/outputs/T_token_propensities_RAND5', width=630, height=500)
display_cohort_tendency_summary('RAND 6', 'tokens', 'p_value', 'voynich_data/outputs/T_token_propensities_RAND6', width=630, height=500)


# -------------------- Cell --------------------
display_cohort_tendency_summary('FIRST', 'tokens', 'p_value', 'voynich_data/outputs/T_token_propensities_FIRST', width=630, height=2000)


# -------------------- Cell --------------------
display_cohort_tendency_summary('TOP', 'tokens', 'bayes', 'voynich_data/outputs/T_token_propensities_TOP', width=630, height=500)




# Z01.5 Extra Analyses.ipynb
print(80*"=")
print(80*"=")
print(f"||  Running Z01.5 Extra Analyses.ipynb")
print(80*"=")
print(80*"=")

# -------------------- Cell --------------------
# Imports and setup
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, HTML
import os

from qlynx.file_utils import load_pkl
from qlynx.stats_utils import *
from qlynx.display_utils import render_html_to_image
from voynichlib.utils import display_voynichese
from qlynx.file_utils import store_pkl, load_pkl


# %reload_ext autoreload
# %autoreload 2
 


# -------------------- Cell --------------------
do_parametric_studies = True
MAX_BAYES = np.exp(10)
MAX_PROPENSITY = 999
THRESHOLDS = {
    'p_value': 0.01,
    'ln_bayes_factor': 5
}
THRESHOLDS['bayes_factor'] = np.exp(THRESHOLDS['ln_bayes_factor'])
reference_cohort = 'MIDDLE'
# smooth = 'laplace'
# smooth = 'laplace'
smooth = None



# -------------------- Cell --------------------
file_path = 'voynich_data/outputs/token_cohort_data.pkl'
token_cohort_data = load_pkl(file_path)

cohorts = token_cohort_data['cohorts']
cohorts_with_randoms = token_cohort_data['cohorts_with_randoms']

corpus_by_c = token_cohort_data['corpus_by_c']
pmfs_by_c = token_cohort_data['pmfs_by_c']
token_ws_by_c = token_cohort_data['token_ws_by_c']

glyph_pmfs_by_c = token_cohort_data['glyph_pmfs_by_c']
glyphs_by_c = token_cohort_data['glyphs_by_c']


# -------------------- Cell --------------------
file_path = 'voynich_data/outputs/token_propensity_dfs.pkl'
token_propensity_dfs = load_pkl(file_path)


# -------------------- Cell --------------------
token_propensity_dfs


# -------------------- Cell --------------------
token_propensity_dfs['TOP']


# -------------------- Cell --------------------
def make_master_table(criteria):
    def meets_criteria(row):
        return row['sig_p_value'] if criteria == 'p_value' else row['sig_BF'] if criteria == 'bayes' else (row['sig_BF']&row['sig_p_value'])
        
    token_dict = {}
    for cohort, propensity_df in token_propensity_dfs.items():
        for token, row in propensity_df.iterrows():
            if token not in token_dict:
                token_dict[token] = {}
            if cohort not in token_dict:
                token_dict[token][cohort] = {}
            token_dict[token][cohort] = 0
            if meets_criteria(row):
                token_dict[token][cohort] = 1 if row['propensity'] > 0 else -1
                # print(f"{token} {cohort}  {token_dict[token][cohort]}")
            pass
        pass
    pass
    # print(token_dict)
    df = pd.DataFrame(columns = ['token', 'TOP', 'FIRST', 'BEFORE', 'AFTER', 'LAST'])
    for token, pos_dict in token_dict.items():
        row_list = [token]
        count_non_zero = 0
        for col in df.columns[1:]:
            if col in pos_dict:
                row_list.append(pos_dict[col])
                count_non_zero += pos_dict[col] != 0
            else:
                row_list.append(0)
        if count_non_zero > 0:
            df.loc[len(df)] = row_list
        pass
    return df
            
df = make_master_table('both')
        # df.loc[len(df)] = [token,
        #                    top_token_length_dict[token],
        #                    N_ref,
        #                    n_ref,
        #                    N_x,
        #                    n_x,
        #                    p_ref,
        #                    p_x,
        #                    p_value,
        #                    verdict_p_value,
        #                    verdict_bayes_factor,
        #                    np.round(propensity,1),
        #                    bayes_factor,
        #                   binom_stat_le,
        #                   binom_stat_gt]


# -------------------- Cell --------------------
df = df.sort_values(by='token')


# -------------------- Cell --------------------
df


# -------------------- Cell --------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_custom_heatmap(df):
    # Set the 'token' column as the index
    df = df.set_index('token')
    
    # Define a custom colormap: red for -1, white for 0, green for 1
    cmap = ListedColormap(['red', 'white', 'green'])
    bounds = [-1, 0, 1, 2]  # Define boundaries for color mapping
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    # Set a larger figure size to accommodate all rows
    plt.figure(figsize=(3, 10))  # You might need to adjust these dimensions based on your specific dataset
    
    # Create a heatmap using seaborn with our custom colormap and normalization
    sns.heatmap(df, annot=False, cmap=cmap, norm=norm, cbar=False, square=False,
                linewidths=0.5, linecolor='black')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha="right")  # Rotate labels and align right for better readability
    

    plt.ion()  # Interactive mode on
    plt.show()
    plt.pause(2)
    plt.close()  



plot_custom_heatmap(df)


# -------------------- Cell --------------------
import pandas as pd

def html_custom_heatmap_to_file(df, filename):
    # Define the color mapping based on value ranges
    def color_for_value(value):
        if value == -1:
            return 'red'
        elif value == 0:
            return 'white'
        elif value == 1:
            return 'green'
        else:
            return 'white'  # Default color

        # Determine the max width needed based on the longest column header name
    # Assuming approximately 1em per character for default font settings
    max_header_length = max(len(str(col)) for col in df.columns)
    column_width = max_header_length -1.5  # Adding some extra space for padding

    # Start the HTML document
    html = '<!DOCTYPE html>\n<html>\n<head>\n'
    html += f'<style>table, th, td {{border: 1px solid black; border-collapse: collapse;}} th, td {{padding: 0px; text-align: center;}} th:not(:first-child), td:not(:first-child) {{min-width: {column_width}em; max-width: {column_width}em;}}</style>\n'
    # html += '<style>table, th, td {border: 1px solid black; border-collapse: collapse;} th, td {padding: 5px; text-align: center;} </style>\n'
    html += '</head>\n<body>\n<h1>High Propensity Tokens by Position</h1>'
    
    # Start the HTML table, including headers at the top
    html += '<table><tr><th></th>'
    for col in df.columns:
        html += f'<th>{col}</th>'
    html += '</tr>'
    
    # Populate the table rows
    for index, row in df.iterrows():
        voynichese_value = display_voynichese(text=index, render=False)
        html += f'<tr><th>{voynichese_value}</th>'
        for col in df.columns:
            color = color_for_value(row[col])
            html += f'<td style="background-color:{color}">&nbsp;</td>'
        html += '</tr>'
    
    # Add headers at the bottom
    html += '<tr><th></th>'
    for col in df.columns:
        html += f'<th>{col}</th>'
    html += '</tr></table>\n'
    
    # End the HTML document
    html += '</body>\n</html>'
    
    # Write the HTML to the specified file
    with open(filename, 'w') as file:
        file.write(html)


# Convert the DataFrame to an HTML file
filename= 'voynich_data/outputs/T_Propensity_Token.html'
html_custom_heatmap_to_file(df.set_index('token'), filename)
print(f"HTML table written to {filename}")


# -------------------- Cell --------------------
import matplotlib.pyplot as plt

def plot_two_letter_combination_histogram(tokens, n, ntop, dir):
    names = {2: 'Two', 3:'Three', 4:'Four', 5:'Five'}
    # Extract and count leading n-letter combinations
    combination_count = {}
    for token in tokens:
        if len(token) > n:
            if dir=='Lead':
                combination = token[:n].lower() # Consider the combination in lowercase to avoid case-sensitive counting
            elif dir == 'Trail':
                combination = token[-n:].lower() # Consider the combination in lowercase to avoid case-sensitive counting
                
            if combination in combination_count:
                combination_count[combination] += 1
            else:
                combination_count[combination] = 1

    # Prepare data for plotting
    combinations = list(combination_count.keys())
    counts = list(combination_count.values())

    # Sort the combinations by count
    sorted_indices = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)
    sorted_combinations = [combinations[i] for i in sorted_indices]
    sorted_counts = [counts[i] for i in sorted_indices]
    a = zip(sorted_combinations, sorted_counts)
    # print([x for x in a])

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.barh(sorted_combinations[:ntop], sorted_counts[:ntop], color='skyblue')
    plt.xlabel('Frequency')
    plt.ylabel(f'{names[n]}-letter Combination')
    plt.title(f'Histogram of {dir}ing {names[n]}-letter Combinations')
    plt.gca().invert_yaxis()  # To display the bar with the highest count at the top

    plt.ion()  # Interactive mode on
    plt.show()
    plt.pause(2)
    plt.close()  

# Example usage




# -------------------- Cell --------------------
(set(''.join(tokens)))


# -------------------- Cell --------------------
tokens = corpus_by_c['ALL'].tokens()


# -------------------- Cell --------------------
plot_two_letter_combination_histogram(tokens, 4, 20, dir='Trail')


# -------------------- Cell --------------------
plot_two_letter_combination_histogram(tokens, 2, 20, dir='Lead')


# -------------------- Cell --------------------
plot_two_letter_combination_histogram(tokens, 3, 20, dir='Lead')


# -------------------- Cell --------------------
plot_two_letter_combination_histogram(tokens, 4, 20, dir='Lead')


# -------------------- Cell --------------------
plot_two_letter_combination_histogram(tokens, 5, 20)


# -------------------- Cell --------------------
def compile_glyph_propensity_df(target_cohort, reference_cohort, p_value_threshold, bayes_threshold):
    top_token_length_dict = get_top_vocabulary_tokens_lengths_dict(reference_cohort)
    df = pd.DataFrame(columns = ['glyph',  'N_ref', 'n_ref', 'N_x', 'n_x', 'p_ref', 'p_x',  'p_value', 'sig_p_value', 'sig_BF', 'propensity', 'bayes', 'binom_stat_le', 'binom_stat_gt'])
    glyphs = set(glyphs_by_c['ALL'])
    
    for glyph in glyphs:
        pmf_ref = glyph_pmfs_by_c[reference_cohort]
        N_ref = pmf_ref.total_count
        n_ref = pmf_ref.count(glyph) if N_ref > 0 else 0
        p_ref = pmf_ref.prob(glyph, smooth=smooth)
        if p_ref == 0:
            continue

        pmf_x = glyph_pmfs_by_c[target_cohort]                        
        N_x = pmf_x.total_count
        n_x = pmf_x.count(glyph)  if N_x > 0 else 0
        p_x = pmf_x.prob(glyph, smooth=smooth)
        
        p_value = calculate_binomial_probability(n_x, N_x, p_ref)

        if n_x == 0:
            bayes_factor = 1/MAX_BAYES
        else:
            bayes_factor = bayes_factor_binomial(n_x, N_x, p_x, p_ref)
            bayes_factor = min(MAX_BAYES, bayes_factor)
        bayes_factor = int(bayes_factor)
        binom_stat_le =  binom.cdf(n_x, N_x, p_ref)        
        binom_stat_gt =  binom.cdf(n_x, N_x, 1. -p_ref)        

        if p_x ==  0.:
            propensity = -MAX_PROPENSITY
        else:
            propensity = p_x/p_ref  if p_ref > 0 else MAX_PROPENSITY
        pass        
            
        verdict_p_value = p_value <= p_value_threshold
        verdict_bayes_factor = bayes_factor >= bayes_threshold

        df.loc[len(df)] = [glyph,
                           N_ref,
                           n_ref,
                           N_x,
                           n_x,
                           p_ref,
                           p_x,
                           p_value,
                           verdict_p_value,
                           verdict_bayes_factor,
                           np.round(propensity,1),
                           bayes_factor,
                          binom_stat_le,
                          binom_stat_gt]
        pass
    df.set_index('glyph', inplace=True)
    pass
    return df


# -------------------- Cell --------------------




