# safety.py
import ast

ALLOWED_NODES = {
    ast.Module, ast.Assign, ast.Expr, ast.Load, ast.Store,
    ast.Name, ast.Constant, ast.Subscript, ast.Slice,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.Call, ast.Attribute, ast.keyword,
    ast.List, ast.Tuple, ast.Dict,
    ast.IfExp,
}

DISALLOWED_NAMES = {"__import__", "open", "exec", "eval", "compile", "globals", "locals", "input"}

DISALLOWED_ATTR_PREFIXES = ("__",)

class UnsafeCodeError(Exception):
    pass

def validate_pandas_code(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise UnsafeCodeError(f"SyntaxError: {e}")

    for node in ast.walk(tree):
        if type(node) not in ALLOWED_NODES:
            raise UnsafeCodeError(f"Disallowed syntax: {type(node).__name__}")

        # Block imports explicitly
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeCodeError("Imports are not allowed")

        # Block suspicious names
        if isinstance(node, ast.Name) and node.id in DISALLOWED_NAMES:
            raise UnsafeCodeError(f"Disallowed name: {node.id}")

        # Block __dunder__ attribute access
        if isinstance(node, ast.Attribute):
            if node.attr.startswith(DISALLOWED_ATTR_PREFIXES):
                raise UnsafeCodeError("Dunder attribute access is not allowed")

    # Require `result` assignment somewhere
    assigns_result = any(isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "result" for t in n.targets
    ) for n in ast.walk(tree))

    if not assigns_result:
        raise UnsafeCodeError("Code must assign to a variable named `result`")