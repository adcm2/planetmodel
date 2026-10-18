"""The public surface: exports resolve, and optional arguments are keyword-only."""
import ast
import dataclasses
import importlib
import inspect
import pkgutil
import tomllib
from pathlib import Path

import planetmodel

ROOT = Path(__file__).resolve().parent.parent


def test_every_export_resolves():
    missing = [n for n in planetmodel.__all__ if not hasattr(planetmodel, n)]
    assert not missing, missing
    assert len(planetmodel.__all__) == len(set(planetmodel.__all__))


def test_version_matches_pyproject():
    with open(ROOT / "pyproject.toml", "rb") as fh:
        declared = tomllib.load(fh)["project"]["version"]
    assert planetmodel.__version__ == declared


# Every parameter with a default is keyword-only, in every function, method
# and constructor of the package; the allowlist below is the agreed set of
# exceptions and is not extended solo.
KEYWORD_ONLY_EXCEPTIONS = set()


def _positional_defaults(fn):
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return []
    return [p.name for p in sig.parameters.values()
            if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            and p.default is not inspect.Parameter.empty]


def _modules():
    """Every module of the package that imports without optional dependencies."""
    mods = [planetmodel]
    for info in pkgutil.walk_packages(planetmodel.__path__, "planetmodel."):
        try:
            mods.append(importlib.import_module(info.name))
        except ImportError:
            pass
    return mods


def _public_callables():
    """Every function and class defined in the package, and every method."""
    seen = set()
    for mod in _modules():
        for name, obj in vars(mod).items():
            if getattr(obj, "__module__", None) != mod.__name__ or obj in seen:
                continue
            seen.add(obj)
            if inspect.isclass(obj):
                yield obj.__name__, obj
                for attr, member in vars(obj).items():
                    if isinstance(member, (staticmethod, classmethod)):
                        member = member.__func__
                    if inspect.isfunction(member):
                        yield f"{obj.__name__}.{attr}", member
            elif inspect.isfunction(obj):
                yield name, obj


def test_optional_arguments_are_keyword_only():
    offenders = {}
    for qualname, fn in _public_callables():
        short = qualname.split(".")[-1]
        if qualname in KEYWORD_ONLY_EXCEPTIONS or short in KEYWORD_ONLY_EXCEPTIONS:
            continue
        bad = _positional_defaults(fn)
        if bad:
            offenders[qualname] = bad
    assert not offenders, f"positional optionals: {offenders}"


# Every dataclass field with a default sits after a `_: KW_ONLY` sentinel, so
# a constructor reads like a signature: required fields, then `*`, then the
# optionals by name.
def test_dataclass_optional_fields_are_keyword_only():
    offenders = [qualname for qualname, obj in _public_callables()
                 if inspect.isclass(obj) and dataclasses.is_dataclass(obj)
                 and any(f.default is not dataclasses.MISSING
                         or f.default_factory is not dataclasses.MISSING
                         for f in dataclasses.fields(obj) if not f.kw_only)]
    assert not offenders, f"positional optional fields: {offenders}"


# Every parameter and every return is annotated, in every function of the
# package, nested functions and dunders included: a signature says what it
# takes and gives.  Read from the source so that modules needing gmsh are
# held to it too.
def _unannotated(tree):
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        a = node.args
        params = a.posonlyargs + a.args + a.kwonlyargs
        missing = [p.arg for p in params
                   if p.annotation is None and p.arg not in ("self", "cls")]
        if a.vararg is not None and a.vararg.annotation is None:
            missing.append("*" + a.vararg.arg)
        if a.kwarg is not None and a.kwarg.annotation is None:
            missing.append("**" + a.kwarg.arg)
        if node.returns is None:
            missing.append("->")
        if missing:
            yield node.lineno, node.name, missing


def test_every_signature_is_annotated():
    root = Path(planetmodel.__file__).parent
    offenders = {}
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        rows = list(_unannotated(tree))
        if rows:
            offenders[str(path.relative_to(root))] = rows
    assert not offenders, f"unannotated signatures: {offenders}"


# Every public module, class, function and method states what it is in a
# docstring.  Read from the source so that modules needing gmsh are held
# to it too; private names, anything nested inside a function, and
# dunders other than __call__ are not held.
def _undocumented(tree):
    def walk(node, in_func):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)):
                public = (child.name == "__call__"
                          or not child.name.startswith("_"))
                if not in_func and public and ast.get_docstring(child) is None:
                    yield child.lineno, child.name
                yield from walk(child, in_func or isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef)))
            else:
                yield from walk(child, in_func)
    yield from walk(tree, False)


def test_every_public_name_has_a_docstring():
    root = Path(planetmodel.__file__).parent
    offenders = {}
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        rows = list(_undocumented(tree))
        if ast.get_docstring(tree) is None:
            rows.insert(0, (1, "(module)"))
        if rows:
            offenders[str(path.relative_to(root))] = rows
    assert not offenders, f"undocumented public names: {offenders}"


# No top-level name needs an extra (standing rule 10): anything that does
# is reached through its module.  Read from the source: a function needs
# an extra when it imports one, when its module imports one at the top,
# or when it calls by name a function that does, in its own module or
# through a name imported from another module of the package.  Calls
# through attributes (`self.f()`, `module.f()`) are not followed.
def _module_name(path, root):
    parts = list(path.relative_to(root.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts), path.name == "__init__.py"


def _resolve(mod, is_package, node):
    """The absolute module an `ImportFrom` names."""
    if not node.level:
        return node.module or ""
    base = mod.split(".") if is_package else mod.split(".")[:-1]
    base = base[:len(base) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


def _needs_extra(extra_modules):
    root = Path(planetmodel.__file__).parent
    extras = set(extra_modules)
    imports_extra = lambda n: (
        isinstance(n, ast.Import) and any(a.name.split(".")[0] in extras
                                          for a in n.names)
        or isinstance(n, ast.ImportFrom) and not n.level
        and (n.module or "").split(".")[0] in extras)
    calls, needs = {}, set()
    for path in sorted(root.rglob("*.py")):
        mod, is_package = _module_name(path, root)
        tree = ast.parse(path.read_text())
        heavy = any(imports_extra(n) for n in tree.body)
        names = {}
        for n in tree.body:
            if isinstance(n, ast.ImportFrom) and n.level:
                for a in n.names:
                    names[a.asname or a.name] = (_resolve(mod, is_package, n), a.name)
        defs = [(n.name, n) for n in tree.body if isinstance(n, ast.FunctionDef)]
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            defs += [(f"{cls.name}.{f.name}", f) for f in cls.body
                     if isinstance(f, ast.FunctionDef)]
        local = {name for name, _ in defs}
        for name, fn in defs:
            key = (mod, name)
            if heavy or any(imports_extra(n) for n in ast.walk(fn)):
                needs.add(key)
            called = {c.func.id for c in ast.walk(fn)
                      if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
            calls[key] = ({(mod, c) for c in called if c in local}
                          | {names[c] for c in called if c in names})
    grown = True
    while grown:
        more = {k for k, callees in calls.items() if callees & needs} - needs
        needs |= more
        grown = bool(more)
    return needs


def test_no_top_level_function_needs_an_extra(extra_modules):
    needs = _needs_extra(extra_modules)
    # the analysis sees what it should: these do need an extra
    assert ("planetmodel.harmonics", "analyse_grid") in needs
    assert ("planetmodel.randomfield.fields", "SphericalGRF.sample_grid") in needs
    assert ("planetmodel.mesh3d.layered", "build_layered_mesh") in needs
    offenders = []
    for name in planetmodel.__all__:
        obj = getattr(planetmodel, name)
        if inspect.isfunction(obj):
            keys = [(obj.__module__, obj.__qualname__)]
        elif inspect.isclass(obj):
            keys = [(obj.__module__, f"{obj.__qualname__}.{m}")
                    for m in ("__init__", "__post_init__", "__new__")]
        else:
            continue
        if any(k in needs for k in keys):
            offenders.append(name)
    assert not offenders, f"top-level names needing an extra: {offenders}"
