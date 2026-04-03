import re


def remove_prefix(s, prefix):
    """Support Python 3.8 and below."""
    if s.startswith(prefix):
        return s[len(prefix) :]
    else:
        return s


def remove_suffix(s, suffix):
    """Support Python 3.8 and below."""
    if s.endswith(suffix):
        return s[: -len(suffix)]
    else:
        return s


def camel_to_snake(name):
    # Split between consecutive uppercase letters followed by an uppercase-lowercase transition
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)

    # Split between lowercase or digit and uppercase letters
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)

    # Split only patterns like 3d, 2d, etc.
    name = re.sub(r"([a-zA-Z])([0-9][a-z]+)", r"\1_\2", name)

    return name.lower()
