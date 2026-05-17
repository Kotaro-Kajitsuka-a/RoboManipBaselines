import re


MATERIAL_OBJECT_PATTERN = re.compile(r"WrenchPredObject[^/\\]*")


def extract_material_object_key(filename):
    match = MATERIAL_OBJECT_PATTERN.search(filename)
    if match is None:
        return None
    return match.group(0)


def build_material_object_key_to_id(filenames):
    #Using set for avoiding the dupulicaed object keys.
    object_keys = sorted(
        {
            object_key
            for filename in filenames
            if (object_key := extract_material_object_key(filename)) is not None
        }
    )
    assert len(object_keys) > 0, "No material object key found in the filenames."
    return {object_key: object_id for object_id, object_key in enumerate(object_keys)}


def get_material_object_id(filename, object_key_to_id):
    object_key = extract_material_object_key(filename)
    assert object_key is not None, f"Cannot find material object key from filename: {filename}"
    return object_key_to_id[object_key]
