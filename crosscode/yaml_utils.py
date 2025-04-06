import torch
import yaml
from crosscode.llms import DTYPE_FROM_STRING

# Create a reverse mapping from PyTorch dtypes to strings
STRING_FROM_DTYPE = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
}

def dtype_representer(dumper: yaml.SafeDumper, data: torch.dtype) -> yaml.ScalarNode:
    """Convert PyTorch dtype to string for YAML serialization"""
    return dumper.represent_str(STRING_FROM_DTYPE[data])

def dtype_constructor(loader: yaml.SafeLoader, node: yaml.ScalarNode) -> torch.dtype:
    """Convert string to PyTorch dtype during YAML deserialization"""
    value = loader.construct_scalar(node)
    if value in DTYPE_FROM_STRING:
        return DTYPE_FROM_STRING[value]
    return value  # Return as is if not a dtype string

# Register the custom representer and constructor
yaml.SafeDumper.add_representer(torch.dtype, dtype_representer)