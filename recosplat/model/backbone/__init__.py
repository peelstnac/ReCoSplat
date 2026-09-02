from recosplat.registry import Registry

from .local_global import ARCfg, BackboneLocalGlobal, BackboneLocalGlobalCfg

BACKBONES: Registry = Registry("backbone")
BACKBONES.register("local_global")(BackboneLocalGlobal)

__all__ = [
    "ARCfg",
    "BACKBONES",
    "BackboneLocalGlobal",
    "BackboneLocalGlobalCfg",
]
