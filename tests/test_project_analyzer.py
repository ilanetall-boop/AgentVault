"""Test ProjectAnalyzer logic on 3 project types.

This script simulates what the VS Code extension's ProjectAnalyzer does
but runs standalone in Python, reproducing the same TS logic.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# --- Fake project builders -------------------------------------------

def create_node_react_project(root: str) -> None:
    """Create a fake Node.js/React project."""
    pkg = {
        "name": "my-react-app",
        "dependencies": {
            "react": "^18.2.0",
            "react-dom": "^18.2.0",
            "next": "^14.0.0",
        },
        "devDependencies": {
            "typescript": "^5.3.0",
            "eslint": "^8.54.0",
            "prettier": "^3.1.0",
            "jest": "^29.7.0",
        },
    }
    Path(root, "package.json").write_text(json.dumps(pkg, indent=2))
    Path(root, "tsconfig.json").write_text('{"compilerOptions":{"strict":true}}')
    Path(root, ".eslintrc.json").write_text('{"extends":"next/core-web-vitals"}')
    Path(root, ".prettierrc").write_text('{"semi":false,"singleQuote":true}')

    # src/ structure
    for sub in ["components", "hooks", "utils", "pages", "services"]:
        d = Path(root, "src", sub)
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.ts").write_text(f"// {sub}")

    # component files
    for comp in ["Button.tsx", "Header.tsx", "Footer.tsx", "Layout.tsx"]:
        (Path(root, "src", "components") / comp).write_text(f"// {comp}")

    Path(root, "public").mkdir(exist_ok=True)
    Path(root, "public", "index.html").write_text("<html></html>")

    # tests
    Path(root, "__tests__").mkdir(exist_ok=True)
    (Path(root, "__tests__") / "app.test.ts").write_text("test('ok', () => {})")

    # Entry point
    (Path(root, "src") / "index.ts").write_text("import React from 'react';")


def create_python_django_project(root: str) -> None:
    """Create a fake Python/Django project."""
    pyproject = """
[project]
name = "mydjango"
version = "1.0.0"
dependencies = ["django>=4.2", "djangorestframework", "celery"]

[tool.ruff]
line-length = 88
select = ["E", "F", "I"]

[tool.ruff.format]
quote-style = "double"

[tool.mypy]
strict = true

[tool.pytest.ini_options]
testpaths = ["tests"]
"""
    Path(root, "pyproject.toml").write_text(pyproject)
    Path(root, "manage.py").write_text("#!/usr/bin/env python\nimport django")
    Path(root, ".editorconfig").write_text("root = true\n[*.py]\nindent_size = 4")

    # Django app structure
    app_dir = Path(root, "myapp")
    app_dir.mkdir(exist_ok=True)
    (app_dir / "__init__.py").write_text("")

    for sub in ["models", "views", "templates", "migrations", "serializers"]:
        d = app_dir / sub
        d.mkdir(exist_ok=True)
        (d / "__init__.py").write_text("")

    # Model files (snake_case)
    for f in ["user_profile.py", "order_item.py", "product_catalog.py"]:
        (app_dir / "models" / f).write_text(f"# {f}")

    for f in ["user_views.py", "order_views.py", "api_views.py"]:
        (app_dir / "views" / f).write_text(f"# {f}")

    # Tests
    tests_dir = Path(root, "tests")
    tests_dir.mkdir(exist_ok=True)
    (tests_dir / "__init__.py").write_text("")
    (tests_dir / "test_models.py").write_text("def test_user(): pass")
    (tests_dir / "test_views.py").write_text("def test_home(): pass")

    # Docker
    Path(root, "Dockerfile").write_text("FROM python:3.11")
    Path(root, "docker-compose.yml").write_text("version: '3'")

    # GitHub Actions
    wf_dir = Path(root, ".github", "workflows")
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / "ci.yml").write_text("name: CI")


# --- Analyzer (reproducing TS logic in Python) ----------------------

IGNORED = {"node_modules", ".git", "dist", "build", "out", "__pycache__",
           ".pytest_cache", ".venv", "venv", ".next", "coverage"}

KNOWN_ROLES = {
    "src": "Source code", "lib": "Libraries", "app": "Application code",
    "api": "API endpoints", "pages": "Page components/routes",
    "components": "UI components", "hooks": "React hooks",
    "utils": "Utilities", "helpers": "Helper functions",
    "services": "Business logic services", "models": "Data models",
    "schemas": "Data schemas", "controllers": "Route controllers",
    "middleware": "Middleware", "routes": "Route definitions",
    "views": "View templates", "templates": "Templates",
    "static": "Static assets", "public": "Public assets",
    "assets": "Assets", "styles": "Stylesheets",
    "tests": "Test files", "test": "Test files", "__tests__": "Test files",
    "spec": "Test specifications", "docs": "Documentation",
    "scripts": "Build/utility scripts", "config": "Configuration",
    "migrations": "Database migrations", "fixtures": "Test fixtures",
    "mocks": "Mock data/services", "benchmarks": "Performance benchmarks",
    "examples": "Usage examples",
}

NAME_ROLES = {
    "core": "Core engine", "api": "API layer", "storage": "Storage/persistence",
    "processing": "Data processing", "sdk": "SDK/client library",
    "cli": "Command-line interface", "ui": "User interface",
    "web": "Web interface", "auth": "Authentication", "config": "Configuration",
    "utils": "Utilities", "helpers": "Helper functions", "models": "Data models",
    "schemas": "Data schemas", "routes": "Route handlers",
    "middleware": "Middleware", "services": "Business logic",
    "handlers": "Request handlers", "plugins": "Plugin system",
    "extensions": "Extensions", "multi_agent": "Multi-agent support",
    "commands": "Command handlers", "capture": "Data capture",
    "context": "Context management", "serializers": "API serializers",
    "migrations": "Database migrations", "templates": "Templates",
    "views": "View templates",
}


def list_dirs(p: str) -> list[str]:
    try:
        return [e.name for e in os.scandir(p)
                if e.is_dir() and e.name not in IGNORED and not e.name.startswith('.')]
    except OSError:
        return []


def infer_dir_role(dir_path: str, dir_name: str) -> str:
    known = NAME_ROLES.get(dir_name.lower(), "")
    try:
        files = [f for f in os.listdir(dir_path)
                 if not f.startswith('.') and f != "__pycache__"]
    except OSError:
        return known or dir_name

    code = [f for f in files
            if f.endswith(('.py', '.ts', '.js', '.tsx', '.jsx'))
            and f not in ('__init__.py', 'index.ts', 'index.js')]

    if code:
        modules = [os.path.splitext(f)[0] for f in code[:6]]
        if known and modules:
            return f"{known} ({', '.join(modules)})"
        if modules:
            return ', '.join(modules)
    return known or dir_name


def analyze_project(root: str) -> dict:
    """Reproduce the TS ProjectAnalyzer logic."""
    name = os.path.basename(root)

    result = {
        "name": name,
        "stack": {"languages": [], "frameworks": [], "tools": [], "infra": []},
        "architecture": {"structure": {}, "patterns": []},
        "conventions": {"linting": {}, "formatting": {}, "naming": "unknown"},
    }

    # === Stack detection ===
    pkg_path = os.path.join(root, "package.json")
    if os.path.exists(pkg_path):
        result["stack"]["languages"].append("JavaScript/TypeScript")
        try:
            pkg = json.loads(Path(pkg_path).read_text())
            deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
            for fw, name in [("react", "React"), ("vue", "Vue"), ("next", "Next.js"),
                             ("express", "Express"), ("fastify", "Fastify")]:
                if fw in deps:
                    result["stack"]["frameworks"].append(name)
            for tool, name in [("typescript", "TypeScript"), ("eslint", "ESLint"),
                               ("prettier", "Prettier"), ("jest", "Jest"), ("vite", "Vite")]:
                if tool in deps:
                    result["stack"]["tools"].append(name)
        except Exception:
            pass

    pyproject_path = os.path.join(root, "pyproject.toml")
    req_path = os.path.join(root, "requirements.txt")
    if os.path.exists(pyproject_path) or os.path.exists(req_path):
        result["stack"]["languages"].append("Python")
        if os.path.exists(pyproject_path):
            content = Path(pyproject_path).read_text().lower()
            for fw, name in [("fastapi", "FastAPI"), ("django", "Django"), ("flask", "Flask")]:
                if fw in content:
                    result["stack"]["frameworks"].append(name)
            for tool, name in [("pytest", "pytest"), ("ruff", "Ruff"), ("mypy", "mypy")]:
                if tool in content:
                    result["stack"]["tools"].append(name)

    if os.path.exists(os.path.join(root, "Dockerfile")):
        result["stack"]["infra"].append("Docker")
    if os.path.exists(os.path.join(root, "docker-compose.yml")):
        result["stack"]["infra"].append("Docker Compose")
    if os.path.exists(os.path.join(root, ".github", "workflows")):
        result["stack"]["infra"].append("GitHub Actions")

    result["stack"]["frameworks"] = list(set(result["stack"]["frameworks"]))
    result["stack"]["tools"] = list(set(result["stack"]["tools"]))

    # === Architecture (2-level) ===
    top_dirs = list_dirs(root)

    for d in top_dirs:
        if d.lower() in KNOWN_ROLES:
            result["architecture"]["structure"][d] = KNOWN_ROLES[d.lower()]

    # Level 2 scan
    proj_name_lower = name.lower().replace("-", "").replace("_", "").replace(" ", "")
    main_dirs = [d for d in top_dirs if (
        d.lower() == proj_name_lower or
        os.path.exists(os.path.join(root, d, "__init__.py")) or
        d.lower() in ("src", "lib", "app")
    )]

    for main_dir in main_dirs:
        main_path = os.path.join(root, main_dir)
        for sub in list_dirs(main_path):
            sub_path = os.path.join(main_path, sub)
            role = infer_dir_role(sub_path, sub)
            result["architecture"]["structure"][f"{main_dir}/{sub}"] = role

    # Patterns
    all_keys = [k.lower() for k in result["architecture"]["structure"]]
    all_vals = [v.lower() for v in result["architecture"]["structure"].values()]

    if any("service" in k for k in all_keys) or any("service" in v for v in all_vals):
        result["architecture"]["patterns"].append("Service Layer")
    if any("component" in k for k in all_keys):
        result["architecture"]["patterns"].append("Component-based")
    if any("api" in k and "route" in k for k in all_keys):
        result["architecture"]["patterns"].append("REST API")

    # === Conventions ===
    for eslint in [".eslintrc.json", ".eslintrc.js", ".eslintrc"]:
        if os.path.exists(os.path.join(root, eslint)):
            result["conventions"]["linting"]["eslint"] = eslint
            break

    if os.path.exists(os.path.join(root, "ruff.toml")):
        result["conventions"]["linting"]["ruff"] = "ruff.toml"

    if os.path.exists(pyproject_path):
        content = Path(pyproject_path).read_text()
        if "[tool.ruff]" in content:
            result["conventions"]["linting"]["ruff"] = "pyproject.toml [tool.ruff]"
        if "[tool.mypy]" in content:
            result["conventions"]["linting"]["mypy"] = "pyproject.toml [tool.mypy]"

    for prettier in [".prettierrc", ".prettierrc.json"]:
        if os.path.exists(os.path.join(root, prettier)):
            result["conventions"]["formatting"]["prettier"] = prettier
            break

    if os.path.exists(pyproject_path):
        content = Path(pyproject_path).read_text()
        if "[tool.ruff.format]" in content:
            result["conventions"]["formatting"]["ruff-format"] = "pyproject.toml [tool.ruff.format]"

    if os.path.exists(os.path.join(root, ".editorconfig")):
        result["conventions"]["formatting"]["editorconfig"] = ".editorconfig"

    return result


def format_context_block(analysis: dict) -> str:
    """Build a context block like contextInjector.ts does."""
    lines = ["=== CodeVault Memory Context ===", ""]

    lines.append(f"PROJECT: {analysis['name']}")
    lines.append("")

    s = analysis["stack"]
    if s["languages"] or s["frameworks"]:
        lines.append("STACK:")
        if s["languages"]:
            lines.append(f"- Languages: {', '.join(s['languages'])}")
        if s["frameworks"]:
            lines.append(f"- Frameworks: {', '.join(s['frameworks'])}")
        if s["tools"]:
            lines.append(f"- Tools: {', '.join(s['tools'])}")
        if s["infra"]:
            lines.append(f"- Infra: {', '.join(s['infra'])}")
        lines.append("")

    a = analysis["architecture"]
    if a["structure"]:
        lines.append("ARCHITECTURE:")
        for folder, desc in list(a["structure"].items())[:12]:
            lines.append(f"- {folder}/: {desc}")
        if a["patterns"]:
            lines.append(f"- Patterns: {', '.join(a['patterns'])}")
        lines.append("")

    c = analysis["conventions"]
    if c["linting"] or c["formatting"] or c["naming"] != "unknown":
        lines.append("CONVENTIONS:")
        if c["naming"] != "unknown":
            lines.append(f"- Naming: {c['naming']}")
        if c["linting"]:
            lines.append(f"- Linting: {', '.join(c['linting'].keys())}")
        if c["formatting"]:
            lines.append(f"- Formatting: {', '.join(c['formatting'].keys())}")
        lines.append("")

    lines.append("================================")
    return "\n".join(lines)


# --- Run tests -------------------------------------------------------

def test_project(name: str, root: str) -> bool:
    print(f"\n{'-'*60}")
    print(f"TEST: {name}")
    print(f"Path: {root}")
    print(f"{'-'*60}")

    analysis = analyze_project(root)
    block = format_context_block(analysis)
    print(block)

    # Assertions
    ok = True

    arch = analysis["architecture"]["structure"]

    if name == "Node.js/React":
        n_dirs = len(arch)
        checks = [
            ("JavaScript/TypeScript" in analysis["stack"]["languages"], "Should detect JS/TS"),
            ("React" in analysis["stack"]["frameworks"], "Should detect React"),
            ("Next.js" in analysis["stack"]["frameworks"], "Should detect Next.js"),
            (any("component" in k.lower() for k in arch), "Should find components/"),
            (any("service" in k.lower() for k in arch), "Should find services/"),
            (n_dirs >= 5, f"Should have >=5 dirs (got {n_dirs})"),
            ("eslint" in analysis["conventions"]["linting"], "Should detect ESLint"),
            ("prettier" in analysis["conventions"]["formatting"], "Should detect Prettier"),
        ]
    elif name == "Python/Django":
        n_dirs = len(arch)
        checks = [
            ("Python" in analysis["stack"]["languages"], "Should detect Python"),
            ("Django" in analysis["stack"]["frameworks"], "Should detect Django"),
            ("Docker" in analysis["stack"]["infra"], "Should detect Docker"),
            (any("model" in k.lower() for k in arch), "Should find models/"),
            (any("view" in k.lower() for k in arch), "Should find views/"),
            (n_dirs >= 4, f"Should have >=4 dirs (got {n_dirs})"),
            ("ruff" in analysis["conventions"]["linting"], "Should detect Ruff"),
            ("mypy" in analysis["conventions"]["linting"], "Should detect mypy"),
            ("editorconfig" in analysis["conventions"]["formatting"], "Should detect EditorConfig"),
            ("ruff-format" in analysis["conventions"]["formatting"], "Should detect ruff-format"),
        ]
    elif name == "Empty project":
        checks = [
            (len(analysis["stack"]["languages"]) == 0, "Should have no languages"),
            (len(analysis["architecture"]["structure"]) == 0, "Should have no dirs"),
        ]
    else:
        checks = []

    for passed, msg in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {msg}")
        if not passed:
            ok = False

    return ok


if __name__ == "__main__":
    tmpdir = tempfile.mkdtemp(prefix="codevault-test-")

    try:
        # Test 1: Node.js/React
        node_dir = os.path.join(tmpdir, "my-react-app")
        os.makedirs(node_dir, exist_ok=True)
        create_node_react_project(node_dir)
        ok1 = test_project("Node.js/React", node_dir)

        # Test 2: Python/Django
        django_dir = os.path.join(tmpdir, "mydjango")
        os.makedirs(django_dir, exist_ok=True)
        create_python_django_project(django_dir)
        ok2 = test_project("Python/Django", django_dir)

        # Test 3: Empty project
        empty_dir = os.path.join(tmpdir, "empty-project")
        os.makedirs(empty_dir, exist_ok=True)
        ok3 = test_project("Empty project", empty_dir)

        # Test 4: AgentVault itself
        agentvault_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ok4 = test_project("AgentVault (real project)", agentvault_root)

        # Summary
        print(f"\n{'='*60}")
        all_ok = all([ok1, ok2, ok3, ok4])
        results = [
            ("Node.js/React", ok1), ("Python/Django", ok2),
            ("Empty", ok3), ("AgentVault", ok4),
        ]
        for name, ok in results:
            print(f"  {'PASS' if ok else 'FAIL'} — {name}")
        print(f"{'='*60}")
        print(f"Overall: {'ALL PASS' if all_ok else 'SOME FAILURES'}")
        sys.exit(0 if all_ok else 1)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
