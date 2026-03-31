models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls

    return decorator


def _load_model_state(model, model_spec):
    if isinstance(model, (tuple, list)):
        if 'sd_g' in model_spec:
            model[0].load_state_dict(model_spec['sd_g'])
        elif 'sd' in model_spec:
            model[0].load_state_dict(model_spec['sd'])

        if len(model) > 1 and model[1] is not None and 'sd_d' in model_spec:
            model[1].load_state_dict(model_spec['sd_d'])
        return model

    if 'sd' in model_spec:
        model.load_state_dict(model_spec['sd'])
    return model


def make(model_spec, load_model=False):
    model = models[model_spec['name']](**model_spec['args'])

    if load_model:
        model = _load_model_state(model, model_spec)

    return model
