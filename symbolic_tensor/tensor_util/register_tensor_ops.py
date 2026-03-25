import torch

def st_pack(tensor):
    from symbolic_tensor.tensor_util.pack_tensor import pack_tensor
    return pack_tensor(tensor)

torch.Tensor.st_pack = st_pack


def st_assign(lvalue, rvalue):
    from symbolic_tensor.tensor_util.assign_tensor import assign_tensor
    return assign_tensor(lvalue, rvalue)

torch.Tensor.st_assign = st_pack

def st_file_paths(tensor):
    from symbolic_tensor.fs_util.get_nested_list_file_pathes import get_nested_list_file_path
    return get_nested_list_file_path(tensor)

torch.Tensor.st_file_paths = st_file_paths