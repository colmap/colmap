def get_matcher(matcher):
    mod = __import__(f"{__name__}.{matcher}", fromlist=[""])
    return getattr(mod, "Model")
