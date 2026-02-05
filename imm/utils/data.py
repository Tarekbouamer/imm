
def extend_keys_with_suffix(data, suffix="0"):
    """Add suffix to all dictionary keys.

    Args:
        data: Dictionary to process
        suffix: Suffix to append to keys

    Returns:
        New dictionary with suffixed keys
    """
    return {k + suffix: v for k, v in data.items()}
