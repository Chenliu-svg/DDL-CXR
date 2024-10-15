from __future__ import absolute_import
from __future__ import print_function

import argparse
import yaml


import pandas as pd
from mimic3benchmark.mimic3csv import *
from mimic3benchmark.preprocessing import add_hcup_ccs_2015_groups, make_phenotype_label_matrix




stays=pd.read_csv('/root/autodl-tmp/ehr/data/root/all_stays.csv')
# stays.to_csv(os.path.join(args.output_path, 'all_stays.csv'), index=False)

diagnoses=pd.read_csv('/root/autodl-tmp/ehr/data/root/all_diagnoses.csv')


phenotypes = add_hcup_ccs_2015_groups(diagnoses, yaml.load(open('/root/dynamic-prediction/mimic4extract/mimic3benchmark/resources/icd_9_10_definitions_2.yaml', 'r'),Loader=yaml.SafeLoader))

# the ground truth phenotype matirx
make_phenotype_label_matrix(phenotypes, stays).to_csv('/root/dynamic-prediction/data/phenotype_labels.csv',
                                                      index=True, quoting=csv.QUOTE_NONNUMERIC)
# phenotypes=pd.read_csv('/root/dynamic-prediction/data/phenotype_labels.csv')
# print(phenotypes.head(5))



