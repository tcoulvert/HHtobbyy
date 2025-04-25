import csv
import json
import re

import pandas as pd
import numpy as np


with open('syst_unc_plots/uncertainties_cat_merged.json', 'r') as jsonfile:
    merged_category_uncs = json.load(jsonfile)

with open('syst_unc_plots/uncertainties_cat.json', 'r') as jsonfile:
    category_uncs = json.load(jsonfile)

variation_postfix = '_percent_diff'

# build the sorted lists for the csv
categories = list(merged_category_uncs.keys())
categories.sort()
samples = list(merged_category_uncs[categories[0]].keys())
samples.sort()
systematics = list(merged_category_uncs[categories[0]][samples[0]].keys())
systematics.sort()
# remove shape systematics
# systematics.remove('Et_dependent_ScaleEB')
# systematics.remove('Et_dependent_ScaleEE')
# systematics.remove('Et_dependent_Smearing')
# remove unnecassary btag systematics
systematics.remove('bTagSF_sys_jes')
systematics.remove('bTagSF_sys_cferr1')
systematics.remove('bTagSF_sys_cferr2')
systematics.remove('bTagSF_sys_lfstats1')
systematics.remove('bTagSF_sys_lfstats2')
systematics.remove('bTagSF_sys_hfstats1')
systematics.remove('bTagSF_sys_hfstats2')
variations = list(merged_category_uncs[categories[0]][samples[0]][systematics[0]].keys())
variations.sort()
variations = [variation[:variation.find(variation_postfix)] for variation in variations]
variations.remove('avg')

unmerged_eras = list(category_uncs[categories[0]].keys())
unmerged_eras.sort()
unmerged_systematics = ['bTagSF_sys_lfstats1', 'bTagSF_sys_lfstats2', 'bTagSF_sys_hfstats1', 'bTagSF_sys_hfstats2']
unmerged_systematics.sort()
extra_systematics = [
    era+'_'+unmrg_syst for unmrg_syst in unmerged_systematics for era in unmerged_eras
]

with open('syst_unc_plots/uncertainties_cat_merged.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    # build the header object
    header = ['sample']
    for category in categories:
        for systematic in systematics+extra_systematics:
            for variation in variations:
                header.append(f'cat{category}_{systematic}_{variation}')
            # header.append(f'cat{category}_{systematic}_avg')
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

                for variation in variations:
                    csv_rows_dict[sample].append(
                        systematic_dict[variation + variation_postfix]
                    )
                # csv_rows_dict[sample].append(
                #     systematic_dict['avg' + variation_postfix]
                # )
        
        for era in unmerged_eras:
            era_dict = category_uncs[category][era]

            for sample in samples:
                if era == 'postEE' and sample == 'VBFToHH':
                    csv_rows_dict[sample].extend(
                        [None]*len(unmerged_systematics)*len(variations)
                    )
                    continue
                sample_dict = era_dict[sample]

                for systematic in unmerged_systematics:
                    systematic_dict = sample_dict[systematic]

                    for variation in variations:
                        csv_rows_dict[sample].append(
                            systematic_dict[variation + variation_postfix]
                        )

    # write the rows
    for sample in samples:
        csvwriter.writerow(csv_rows_dict[sample])


small_pct = 0.0125
close_ratio = 0.5

df = pd.read_csv('syst_unc_plots/uncertainties_cat_merged.csv')

averaged_columns = [col[:-len('_up')] for col in df.columns if re.search('up', col) is not None]
averaged_df = pd.DataFrame(
    [[0.]*len(averaged_columns)]*len(samples),
    columns=averaged_columns
)

averaged_merged_columns = [col[len('cat0_'):-len('_up')] for col in df.columns if re.search('cat0', col) is not None and re.search('_up', col) is not None]
averaged_merged_df = pd.DataFrame(
    [[0.]*len(averaged_merged_columns)]*len(samples),
    columns=averaged_merged_columns
)

for syst_type in averaged_merged_columns:
    syst_sub_df = df.loc[:, [col for col in df.columns if re.search(syst_type, col) is not None]]

    for idx, sample in enumerate(samples):
        sample_up = syst_sub_df.loc[idx, [col for col in syst_sub_df.columns if re.search('up', col) is not None]].to_numpy()
        sample_down = syst_sub_df.loc[idx, [col for col in syst_sub_df.columns if re.search('down', col) is not None]].to_numpy()
        
        avg_sample = np.zeros_like(sample_up)
        for i in range(len(sample_up)):
            if sample_up[i] == 0 and sample_down[i] != 0:
                avg_sample[i] = sample_down[i]
            elif sample_down[i] == 0 and sample_up[i] != 0:
                avg_sample[i] = sample_up[i]
            else:
                avg_sample[i] = 0.5 * (
                    np.abs(sample_up[i])
                    + np.abs(sample_down[i])
                )

        merge_bool = (
            np.all(avg_sample) < small_pct
        ) or (
            np.all([
                [
                    (
                        (avg_sample[i] / avg_sample[j]) > close_ratio 
                        and (avg_sample[i] / avg_sample[j]) < 1+close_ratio
                    ) for j in range(i+1, len(avg_sample))
                ] 
                for i in range(len(avg_sample))
            ])
        )

        averaged_merged_df.loc[idx, syst_type] = np.mean(avg_sample)
        averaged_df.loc[idx, [col for col in averaged_df.columns if re.search(syst_type, col) is not None]] = avg_sample

averaged_df.insert(0, 'sample', df.loc[:, 'sample'])
averaged_merged_df.insert(0, 'sample', df.loc[:, 'sample'])
averaged_df.to_csv('syst_unc_plots/masked_avg_uncs.csv', index=False)
averaged_merged_df.to_csv('syst_unc_plots/masked_avgMerge_uncs.csv', index=False)
        



