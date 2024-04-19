def features_to_group_map(features,attr):
    attr_map = {}
    for feature in features:
        value = getattr(feature,attr)
        if value not in attr_map:
            attr_map[value] = []
        attr_map[value].append(feature)
    return attr_map

