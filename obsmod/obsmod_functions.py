"""
Module of functions for LO/obsmod
"""

source_dict = {
    'all':['ecology_nc', 'kc_whidbeyBasin', 'nceiSalish'],
    
}


def parse_sources_arg(sources_arg):
    """
    Parse the -sources argument.

    Accepted forms:
    - Named key in source_dict, e.g. "all"
    - Single source, e.g. "ecology_nc"
    - Comma-separated list, e.g. "ecology_nc,kc_whidbeyBasin"
    """
    if sources_arg in source_dict:
        return source_dict[sources_arg]

    source_list = [s.strip() for s in sources_arg.split(',') if s.strip()]
    if len(source_list) == 0:
        raise ValueError('No sources provided in -sources argument.')
    return source_list