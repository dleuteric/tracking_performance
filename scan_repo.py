# dev/scan_repo.py
from __future__ import annotations
import os, ast, sys, textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).parent.name=="dev") else Path.cwd()

def list_py_files(root: Path):
    for p, _, files in os.walk(root):
        # ignora venv, cache, hidden, .git
        if any(x in p for x in (".venv", "__pycache__", ".git", ".idea")):
            continue
        for f in files:
            if f.endswith(".py"):
                yield Path(p)/f

def tree(root: Path) -> str:
    lines=[]
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".venv","__pycache__", ".git", ".idea")]
        rel = Path(dirpath).relative_to(root)
        indent = "  " * (len(rel.parts))
        if rel != Path("."):
            lines.append(f"{indent}{rel.name}/")
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                lines.append(f"{indent}  {fn}")
    return "\n".join(lines)

def parse_file(path: Path):
    try:
        src = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(src, filename=str(path))
    except Exception as e:
        return {"error": str(e)}
    funcs, classes, mains = [], {}, False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            funcs.append(node.name)
        elif isinstance(node, ast.ClassDef):
            methods=[n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            classes[node.name]=methods
        elif isinstance(node, ast.If):
            # detect: if __name__ == "__main__":
            test = ast.dump(node.test)
            if "__name__" in test and "__main__" in test:
                mains = True
    return {"funcs": funcs, "classes": classes, "has_main": mains}

def main():
    root = ROOT
    print(f"[ROOT] {root}\n")
    print("[TREE]")
    print(tree(root))
    print("\n[INDEX]")
    for py in sorted(list_py_files(root)):
        rel = py.relative_to(root)
        info = parse_file(py)
        if "error" in info:
            print(f"- {rel}  (parse ERROR: {info['error']})")
            continue
        print(f"- {rel}")
        if info["funcs"]:
            print("    funcs:", ", ".join(info["funcs"]))
        if info["classes"]:
            for cls, methods in info["classes"].items():
                mlist = ", ".join(methods) if methods else "-"
                print(f"    class {cls}: {mlist}")
        if info["has_main"]:
            print("    entrypoint: __main__")
    print("\n[HINT] To narrow scope: python dev/scan_repo.py | sed -n '/\\[INDEX\\]/,$p'")

if __name__ == "__main__":
    main()