"""
This file provides the implementation of pie charts to visualize how
the various used dataset are built up.
"""

import matplotlib.pyplot as plt

# statistics of OG dataset
total_tweets_OG = 4335
sarcastic_tweets_OG = 867
non_sarcastic_tweets_OG = total_tweets_OG - sarcastic_tweets_OG

# statistics of balanced dataset
total_tweets_BD = 6936
sarcastic_tweets_BD = 3468
non_sarcastic_tweets_BD = total_tweets_BD - sarcastic_tweets_BD

# statistics of imbalanced dataset
total_tweets_ID = 8670
sarcastic_tweets_ID = 867*2
non_sarcastic_tweets_ID = total_tweets_ID - sarcastic_tweets_ID

# statistics of test dataset
total_tweets_test = 1400
sarcastic_tweets_test = 200
non_sarcastic_tweets_test = total_tweets_test - sarcastic_tweets_test

# colors for pie charts
colors = ['lightcoral', 'lightcoral', 'lightblue', 'lightblue']

# sizes for each section of pie chart
sizes_OG = [non_sarcastic_tweets_OG, 
            sarcastic_tweets_OG]

sizes_balanced = [non_sarcastic_tweets_OG, 
                  sarcastic_tweets_OG, 
                  non_sarcastic_tweets_BD, 
                  sarcastic_tweets_BD]

sizes_imbalanced = [non_sarcastic_tweets_OG, 
                    sarcastic_tweets_OG, 
                    non_sarcastic_tweets_ID, 
                    sarcastic_tweets_ID]

sizes_test = [non_sarcastic_tweets_test, 
              sarcastic_tweets_test]

# labels for each section for each pie chart
labels_OG = ['Non-sarcastic (OG)\n3468 Tweets', 
             'Sarcastic (OG)\n867 Tweets']

labels_balanced = ['Non-sarcastic (OG)\n3468 Tweets', 
                   'Sarcastic (OG)\n867 Tweets', 
                   'Non-sarcastic (GV)\n 0 Tweets', 
                   'Sarcastic (VG)\n867 Tweets']

labels_imbalanced = ['Non-sarcastic (OG)\n3468 Tweets', 
                     'Sarcastic (OG)\n867 Tweets', 
                     'Non-sarcastic (GV)\n3468 Tweets', 
                     'Sarcastic (GV)\n867 Tweets']

labels_test = ['Non-sarcastic (OG)\n1200 Tweets', 
               'Sarcastic (OG)\n200 Tweets']

# patterns for sarcastic sections
hatches = ['', '////', '', '////'] 

# subplots: 1 row, 4 columns
fig, axes = plt.subplots(1, 4, figsize=(16, 4))  

# plot OG dataset pie chart
patches, texts, autotexts = axes[0].pie(sizes_OG, labels=labels_OG, colors=colors, autopct='%1.1f%%', startangle=140, wedgeprops={'hatch': '/', 'linewidth': 2.5, 'edgecolor': 'white'}, textprops={'fontsize': 10})
for patch, hatch in zip(patches, hatches):
    patch.set_hatch(hatch)
axes[0].set_title('OG Dataset\n4335 Tweets', y=1.0)
axes[0].axis('equal')

# plot balanced dataset pie chart
patches, texts, autotexts = axes[1].pie(sizes_balanced, labels=labels_balanced, colors=colors, autopct='%1.1f%%', startangle=140, wedgeprops={'hatch': '/', 'linewidth': 2.5, 'edgecolor': 'white'}, textprops={'fontsize': 10})
for patch, hatch in zip(patches, hatches):
    patch.set_hatch(hatch)
axes[1].set_title('Balanced Dataset\n6936 Tweets', y=1.0)
axes[1].axis('equal')

# plot inbalanced dataset pie chart
patches, texts, autotexts = axes[2].pie(sizes_imbalanced, labels=labels_imbalanced, colors=colors, autopct='%1.1f%%', startangle=140, wedgeprops={'hatch': '/', 'linewidth': 2.5, 'edgecolor': 'white'}, textprops={'fontsize': 10})
for patch, hatch in zip(patches, hatches):
    patch.set_hatch(hatch)
axes[2].set_title('Imbalanced Dataset\n8670 Tweets', y=1.0)
axes[2].axis('equal')

# plot test dataset pie chart
patches, texts, autotexts = axes[3].pie(sizes_test, labels=labels_test, colors=colors, autopct='%1.1f%%', startangle=140, wedgeprops={'hatch': '/', 'linewidth': 2.5, 'edgecolor': 'white'}, textprops={'fontsize': 10})
for patch, hatch in zip(patches, hatches):
    patch.set_hatch(hatch)
axes[3].set_title('Test Dataset\1400 Tweets', y=1.0)
axes[3].axis('equal')

plt.tight_layout()
plt.show()
