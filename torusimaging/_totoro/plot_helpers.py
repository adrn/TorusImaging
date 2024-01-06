def elem_to_label(elem, dollar=True):
    """
    Convert an element abundance ratio column name to a Latex label.

    Parameters
    ----------
    elem : str
        The input column name. This should be a string like 'MG_FE' or 'mg_fe'.
    dollar : bool (optional)
        Controls whether or not to wrap the output label in inline math dollar
        signs, $.

    Returns
    -------
    elem_label : str
        The element abundance ratio reformatted as a column name.
    """

    try:
        num, den = elem.split("_")
    except ValueError as e:
        raise ValueError(
            f"Invalid element column name '{elem}' - expected a string like X_Y"
        ) from e

    labels = []
    for part in map(str.lower, [num, den]):
        if len(part) > 2:  # Assume it's a greek letter...
            label = "\\" + part
        elif part == "m":
            label = r"{\rm M}"
        else:
            label = r"{\rm " + part.title() + " }"
        labels.append(label)

    label = f"[{labels[0]} / {labels[1]}]"

    if dollar:
        return f"${label}$"
    else:
        return label
