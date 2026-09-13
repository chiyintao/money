"""The architecture, enforced rather than described.

`app/` used to be a flat namespace of 79 modules, and the only thing keeping the layers
apart was that nobody had yet imported the wrong way. Two real defects came out of that:
the venue layer imported the order layer for two plain value objects, and the serving
layer reached back UP into the model layer for one scoring function. Both were patched at
the call site with a function-local import, which is the usual way a cycle gets hidden
rather than fixed.

The contract, bottom-up:

    core      vocabulary and mechanism -- imports no app module at all
    storage   durability
    market    the venue
    features  the feature contract, shared verbatim by training and serving
    trading   order mechanics, sizing, risk
    backtest  replay and state projection
    models    fitting, evaluation, artifacts
    strategy  model output -> trade decision
    ops, web  observability and read models

A package may import only packages EARLIER in that list. `main` and `runtime_context` are
the composition root and are exempt because wiring is their whole job.

Three separate things are checked, because they fail differently:

* a module-level import of a higher layer -- a true layering violation
* a cycle even when every edge points down -- still an unimportable knot
* a function-local import of a higher layer -- a hidden violation, since it does not
  break at import time and so survives review
"""
import ast
import os

LAYERS = ['core', 'storage', 'market', 'features', 'trading', 'backtest', 'models',
          'strategy', 'ops', 'web']
INDEX = {name: i for i, name in enumerate(LAYERS)}

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.join(APP, 'app')
ROOT_MODULES = ('main', 'runtime_context')


def package_of(path):
    """The layer a file belongs to, or "_root" for the composition root."""
    parts = os.path.relpath(path, APP_DIR).replace(os.sep, '/').split('/')
    return '_root' if len(parts) == 1 else parts[0]


def app_files():
    for dirpath, _, names in os.walk(APP_DIR):
        for name in sorted(names):
            if name.endswith('.py'):
                yield os.path.join(dirpath, name)


def _main_guard_lines(tree):
    """Line numbers inside an `if __name__ == '__main__'` block.

    That block is an entry point, not part of the library: it runs only when the module is
    executed directly, never when it is imported. Imports there are composition, exactly
    like main.py, and are not library dependencies.
    """
    lines = set()
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        text = ast.dump(node.test).lower()
        if "'__main__'" not in text and '"__main__"' not in text:
            continue
        for child in ast.walk(node):
            end = getattr(child, 'end_lineno', None) or getattr(child, 'lineno', None)
            start = getattr(child, 'lineno', None)
            if start and end:
                lines.update(range(start, end + 1))
    return lines


def imported_layers(path, module_level_only):
    """Layers this file imports from.

    ``module_level_only`` restricts to imports at the top of the module: a function-local
    import creates no import-time dependency. Either way, `__main__`-guard code is ignored.
    """
    tree = ast.parse(open(path, encoding='utf-8').read())
    guarded = _main_guard_lines(tree)
    nodes = tree.body if module_level_only else list(ast.walk(tree))
    current = package_of(path)
    base = ['app'] if current == '_root' else ['app', current]
    found = set()
    for node in nodes:
        if not isinstance(node, ast.ImportFrom) or not node.level:
            continue
        if node.lineno in guarded:
            continue
        up = node.level - 1
        prefix = base[:len(base) - up] if up else base
        rest = (node.module or '').split('.') if node.module else []
        full = prefix + rest
        if len(full) >= 2 and full[0] == 'app':
            found.add(full[1])
    return found


def _deferred_by_file():
    """{relative file: {layer}} for imports that are NOT at module level."""
    out = {}
    for path in app_files():
        current = package_of(path)
        deferred = imported_layers(path, module_level_only=False) - \
            imported_layers(path, module_level_only=True) - {current}
        if deferred:
            out[os.path.relpath(path, APP).replace(os.sep, '/')] = deferred
    return out


def test_no_module_level_import_of_a_higher_layer():
    """The primary contract: the dependency graph points one way."""
    violations = []
    for path in app_files():
        current = package_of(path)
        if current == '_root':
            continue
        for target in sorted(imported_layers(path, module_level_only=True)):
            if target in ROOT_MODULES:
                violations.append('%s imports the composition root'
                                  % os.path.relpath(path, APP))
            elif target in INDEX and INDEX[target] > INDEX[current]:
                violations.append('%s: %s imports %s, which is above it'
                                  % (os.path.relpath(path, APP), current, target))
    assert not violations, 'module-level layer violations:\n  ' + '\n  '.join(violations)


def test_no_deferred_import_of_a_higher_layer():
    """A function-local import hides a violation instead of removing it."""
    violations = []
    for name, targets in sorted(_deferred_by_file().items()):
        current = name.split('/')[0]
        for target in sorted(targets):
            if target in ROOT_MODULES:
                violations.append('%s defers to the composition root' % name)
            elif target in INDEX and current in INDEX and INDEX[target] > INDEX[current]:
                violations.append('%s: %s defers to %s, which is above it'
                                  % (name, current, target))
    assert not violations, ('upward deferred imports -- wire these properly:\n  '
                            + '\n  '.join(violations))


def test_there_is_no_package_cycle():
    """Downward-only edges can still close a loop through a third package."""
    graph = {name: set() for name in LAYERS}
    for path in app_files():
        current = package_of(path)
        if current not in graph:
            continue
        for target in imported_layers(path, module_level_only=False):
            if target in graph and target != current:
                graph[current].add(target)
    colour, stack, cycles = {}, [], []

    def walk(node):
        colour[node] = 1
        stack.append(node)
        for nxt in sorted(graph[node]):
            if colour.get(nxt) == 1:
                cycles.append(stack[stack.index(nxt):] + [nxt])
            elif colour.get(nxt, 0) == 0:
                walk(nxt)
        stack.pop()
        colour[node] = 2

    for name in LAYERS:
        if colour.get(name, 0) == 0:
            walk(name)
    assert not cycles, 'package cycles: ' + '; '.join(' -> '.join(c) for c in cycles)


def test_core_depends_on_nothing():
    """Every layer may import core, so core may import none of them."""
    offenders = []
    for path in app_files():
        if package_of(path) != 'core':
            continue
        for target in sorted(imported_layers(path, module_level_only=False) - {'core'}):
            offenders.append('%s -> %s' % (os.path.relpath(path, APP), target))
    assert not offenders, 'core must not import app modules: ' + ', '.join(offenders)


def test_deferred_imports_stay_bounded():
    """Each deferred import is a cycle waiting to be written. Cap the population.

    The survivors are genuine lazy loads: an optional store, a CLI-only model runtime, a
    heavy dependency that is not always installed. If this number grows, a new one was
    added for convenience rather than necessity.
    """
    deferred = _deferred_by_file()
    total = sum(len(v) for v in deferred.values())
    assert total <= 30, ('%d deferred imports; the cap is 30.\n' % total
                         + '\n'.join('  %s -> %s' % (k, sorted(v))
                                      for k, v in sorted(deferred.items())))
