import os
import pathlib

def print_tree(startpath, prefix=""):
    items = sorted(os.listdir(startpath))
    for i, name in enumerate(items):
        path = os.path.join(startpath, name)
        connector = "└── " if i == len(items) - 1 else "├── "
        print(prefix + connector + name)
        if os.path.isdir(path):
            extension = "    " if i == len(items) - 1 else "│   "
            print_tree(path, prefix + extension)

# trova la root a seconda del sistema operativo
root = pathlib.Path(os.getcwd()).anchor
print(f"Root: {root}")
print_tree(root)
