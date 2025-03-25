try:
    from .explore import make_lp
except ModuleNotFoundError as mnfe:
    import sys
    print("Unable to import lp dependencies, to use install this repository with `pip install feature-inspect[lp_inspect]` or `pip install feature-inspect[all]`",
          file=sys.stderr)

__all__ = ["make_lp"]
