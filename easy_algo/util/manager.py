class ModelFactory(object):
    _registry = {}
    _model_type = {}

    @staticmethod
    def register(name, clazz, model_type=None, default_params=None):
        if name in ModelFactory._registry:
            raise TypeError("Duplicate operation named {}".format(name))

        # Define a function to handle the creation with default parameters
        def create_with_defaults(*args, **kwargs):
            params = default_params.copy() if default_params else {}
            params.update(kwargs)

            return clazz(*args, **params)

        ModelFactory._model_type[name] = model_type
        ModelFactory._registry[name] = create_with_defaults

    @staticmethod
    def contains(name):
        return name in ModelFactory._registry

    @staticmethod
    def list():
        return list(ModelFactory._registry.keys())

    @staticmethod
    def get_model_type(name):
        if name in ModelFactory._model_type:
            return ModelFactory._model_type[name]
        raise TypeError("Unknown model type {}".format(name))

    @staticmethod
    def create(name, *args, **kwargs):
        if name not in ModelFactory._registry:
            raise ValueError("Unsupported feature operation name")

        # Use the registered function to create the instance
        return ModelFactory._registry[name](*args, **kwargs)
