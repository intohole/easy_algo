def covertTo(obj1, obj2, fields=None):
    if obj1 is None or obj2 is None:
        return

    # 如果fields是None，则获取obj2的所有属性名称
    if fields is None:
        fields = dir(obj2)

    for field in fields:
        # 检查obj1是否有该字段
        if hasattr(obj1, field):
            # 检查obj2是否有该字段，如果有，则更新
            if hasattr(obj2, field):
                setattr(obj2, field, getattr(obj1, field))


def covertFromExclude(obj1, obj2, exclude_fields=None):
    if obj1 is None or obj2 is None:
        return

    # 如果exclude_fields是None，则获取obj1的所有属性名称
    if exclude_fields is None:
        exclude_fields = []

    # 获取obj1的所有属性名称
    fields = dir(obj1)

    for field in fields:
        # 检查该字段是否不在排除列表中
        if field not in exclude_fields:
            # 检查obj2是否有该字段，如果有，则更新
            if hasattr(obj2, field):
                setattr(obj2, field, getattr(obj1, field))
