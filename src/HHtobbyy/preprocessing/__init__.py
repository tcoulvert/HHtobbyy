from .condor_preprocess import LPCVanillaSubmitter

from .preprocess import add_basic_info

# Identity builder function, simply returns DF as-is
def indentity(df, filepath, prefactor): return df

from .resolved_preprocessing import add_vars_resolvedMLP