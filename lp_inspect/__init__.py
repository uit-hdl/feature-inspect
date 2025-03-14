try:
    from .explore import lp_eval
except ModuleNotFoundError as mnfe:
    import sys
    print("Unable to import lp dependencies, to use install this repository with `pip install feature-inspect[lp_inspect]`",
          file=sys.stderr)

__all__ = ["lp_eval"]
