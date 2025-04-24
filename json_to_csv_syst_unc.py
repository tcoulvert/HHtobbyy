import csv
import json
import re

import pandas as pd
import numpy as np


with open('syst_unc_plots/uncertainties_cat_merged.json', 'r') as jsonfile:
    merged_category_uncs = json.load(jsonfile)

variation_postfix = '_percent_diff'

# build the sorted lists for the csv
categories = list(merged_category_uncs.keys())
categories.sort()
samples = list(merged_category_uncs[categories[0]].keys())
samples.sort()
systematics = list(merged_category_uncs[categories[0]][samples[0]].keys())
systematics.sort()
variations = list(merged_category_uncs[categories[0]][samples[0]][systematics[0]].keys())
variations.sort()
variations = [variation[:variation.find(variation_postfix)] for variation in variations]

with open('syst_unc_plots/uncertainties_cat_merged.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    # build the header object
    header = ['sample']
    for category in categories:
        for systematic in systematics:
            # for variation in variations:
            #     header.append(f'cat{category}_{systematic}_{variation}')
            header.append(f'cat{category}_{systematic}_avg')
    csvwriter.writerow(header)

    # build the dict for storing the rows as they will be input to the csv
    csv_rows_dict = {
        sample: [sample] for sample in samples
    }
    for category in categories:
        category_dict = merged_category_uncs[category]

        for sample in samples:
            sample_dict = category_dict[sample]

            for systematic in systematics:
                systematic_dict = sample_dict[systematic]

                # for variation in variations:
                #     csv_rows_dict[sample].append(
                #         systematic_dict[variation + variation_postfix]
                #     )
                csv_rows_dict[sample].append(
                    systematic_dict['avg' + variation_postfix]
                )

    # write the rows
    for sample in samples:
        csvwriter.writerow(csv_rows_dict[sample])


df = pd.read_csv('syst_unc_plots/uncertainties_cat_merged.csv')

per_category_columns = [col[len('cat0_'):] for col in df.columns if re.search('cat0', col) is not None]

to_merge_df = pd.DataFrame(
    [['separated']*len(per_category_columns)]*len(samples),
    columns=per_category_columns
)
for syst_type in per_category_columns:
    syst_sub_df = df.loc[:, [col for col in df.columns if re.search(syst_type, col) is not None]]

    for idx, sample in enumerate(samples):
        sample_np = syst_sub_df.loc[idx].to_numpy()
        

        if (
            np.mean(sample_np) < 0.02
         ) or (
            np.mean(sample_np) < 0.05
            and np.std(sample_np) < 0.01
         ):
            to_merge_df.loc[idx, syst_type] = np.mean(sample_np)
            df.loc[idx, [col for col in df.columns if re.search(syst_type, col) is not None]] = 'merged'

print(len(df.columns))
print(len(to_merge_df.columns))
df = pd.merge(df, to_merge_df, 'outer')
print(len(df.columns))


