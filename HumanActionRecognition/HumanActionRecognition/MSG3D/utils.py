from importlib import import_module


def import_class(name):
    components = name.split('.')
    #mod = __import__(components[0])
    mod = import_module(components[0] + "." + components[1] + "." + components[2], __package__)
    for comp in components[3:]:
        mod = getattr(mod, comp, None)
    return mod


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)