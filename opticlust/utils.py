def validate_resolutions(columns):
    """
    Validate the naming scheme of the resolution columns.
    Column names should be in the shape "[method]_res_[res]".

    :param columns: list of adata.obs column names.

    :return: tuple with string "leiden" or "louvain",
    and a list of resolutions as floats.
    """
    error = ValueError("Column names must be in the shape '[method]_res_[res]'")
    if columns[0].count("_") != 2:
        raise error
    method_clustering, res, _ = columns[0].split("_")
    if method_clustering not in ["leiden", "louvain"] or res != "res":
        raise error
    try:
        resolutions = [float(c.rsplit("_", 1)[1]) for c in columns]
    except ValueError:
        raise error

    return method_clustering, resolutions
