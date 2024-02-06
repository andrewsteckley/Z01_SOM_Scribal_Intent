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


