import argparse
import glob
import logging
import os

from resolved_preprocessing import  add_vars_resolved
from boosted_preprocessing import add_vars_boosted
from preprocessing_utils import match_sample


################################


################################


logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description="Preprocess data to add necessary BDT variables.")
parser.add_argument(
    "--sim_era_filepaths", 
    default=None,
    help="Full filepath(s) (separated with \',\') on LPC for MC era(s)"
)

################################


def run_preprocessing():

if __name__ == '__main__':
    args = parser.parse_args()

    sim_eras = {
        os.path.join(era, '') for era in args.sim_era_filepaths
    } if args.sim_era_filepaths is not None else None
    make_mc(sim_eras=sim_eras, output=args.output_dirpath)

    data_eras = {
        os.path.join(era, '') for era in args.data_era_filepaths
    } if args.data_era_filepaths is not None else None
    make_data(data_eras=data_eras, output=args.output_dirpath)
