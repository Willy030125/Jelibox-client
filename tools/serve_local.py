#!/usr/bin/env python3
"""
Install THIS copy of Jelibox (with your own changes) on any computer with the usual one-line command.

    python tools/serve_local.py                 # pack this folder and serve it on port 8000 (Ctrl+C stops)
    python tools/serve_local.py --port 8080 --bind 0.0.0.0
    python tools/serve_local.py --pack-only     # only write dist/ (jelibox-local.zip, .tar.gz, install.ps1, install.sh)

While it runs, on the computer that should get Jelibox (this one: use localhost; another one on the network: use the
address printed below):

    Windows (PowerShell):   irm http://localhost:8000/install.ps1 | iex
    Linux / macOS (bash):   curl -fsSL http://localhost:8000/install.sh | bash

The installer is the normal one (install.ps1 / install.sh); the server only adds the address of the packed archive in
front of it, so nothing is downloaded from GitHub. Running the command again updates an existing install in place and
never touches its datasets, models or configs. What is packed: the source code, as in a GitHub release, without your
data (datasetsInput, models, configs, export folders, YOLO/VOC datasets), the virtual environment or caches.
"""
import argparse
import datetime
import http.server
import io
import os
import re
import socket
import tarfile
import zipfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DIST = os.path.join(REPO, "dist")
ZIP_NAME, TGZ_NAME = "jelibox-local.zip", "jelibox-local.tar.gz"

# Folders that hold your own data or are generated on each machine. Only their .gitignore placeholder is packed.
DATA_DIRS = {"jelibox", "venv", "exportTools", "datasetsInput", "configs", "export dataset", "export model",
             "YOLOdataset", "models", "vocdataset"}
SKIP_DIRS = {"__pycache__", ".git", ".python", ".uv", "dist", "scratch", ".pytest_cache", ".mypy_cache"}
SKIP_FILES = {"Jelibox.desktop", "Jelibox-launcher.sh", ".install-yes", ".jelibox-version", ".jelibox-ready"}
SKIP_SUFFIX = (".pyc", ".lnk")


def read_version():
    with open(os.path.join(REPO, "utils", "__init__.py"), encoding="utf-8") as f:
        m = re.search(r'^__version__\s*=\s*"([^"]+)"', f.read(), re.M)
    return m.group(1) if m else "0"


def pack_list():
    """Relative paths (posix style) of the files that go into the archive."""
    out = []
    for dirpath, dirs, files in os.walk(REPO):
        rel_dir = os.path.relpath(dirpath, REPO).replace(os.sep, "/")
        top = "" if rel_dir == "." else rel_dir.split("/")[0]
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(files):
            if name in SKIP_FILES or name.endswith(SKIP_SUFFIX):
                continue
            if top in DATA_DIRS and not (rel_dir == top and name == ".gitignore"):
                continue
            out.append(name if rel_dir == "." else f"{rel_dir}/{name}")
    return out


def pack(label):
    """Write dist/jelibox-local.zip and .tar.gz; both hold one top folder named Jelibox-client-<label>."""
    os.makedirs(DIST, exist_ok=True)
    top, files = f"Jelibox-client-{label}", pack_list()
    zpath, tpath = os.path.join(DIST, ZIP_NAME), os.path.join(DIST, TGZ_NAME)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel in files:
            z.write(os.path.join(REPO, rel), f"{top}/{rel}")
    with tarfile.open(tpath, "w:gz") as t:
        for rel in files:
            ti = t.gettarinfo(os.path.join(REPO, rel), f"{top}/{rel}")
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            if rel.endswith((".sh", ".bash")):
                ti.mode = 0o755
            with open(os.path.join(REPO, rel), "rb") as f:
                t.addfile(ti, f)
    return top, len(files), zpath, tpath


def lan_address():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return None


def installer_with_url(name, base):
    """install.ps1 / install.sh as they are in this folder, with the archive address put in front (unless already set)."""
    with open(os.path.join(REPO, name), "rb") as f:
        data = f.read()
    if name.endswith(".ps1"):
        head = (f"if (-not $env:JELIBOX_URL) {{ $env:JELIBOX_URL = '{base}/{ZIP_NAME}' }}\r\n").encode()
        return head + data
    first, _, rest = data.partition(b"\n")             # keep the #! line first
    head = (f'export JELIBOX_URL="${{JELIBOX_URL:-{base}/{TGZ_NAME}}}"\n').encode()
    return first + b"\n" + head + rest


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=DIST, **kw)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/install.ps1", "/install.sh"):
            base = f"http://{self.headers.get('Host') or 'localhost'}"
            body = installer_with_url(path[1:], base)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, fmt, *args):
        print("  " + fmt % args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--bind", default="0.0.0.0", help="address to listen on (127.0.0.1 = only this computer)")
    ap.add_argument("--pack-only", action="store_true", help="write dist/ and stop")
    a = ap.parse_args()

    label = f"{read_version()}-local.{datetime.datetime.now():%Y%m%d%H%M}"
    top, n, zpath, tpath = pack(label)
    print(f"Packed {n} files as {top}")
    for p in (zpath, tpath):
        print(f"  {os.path.relpath(p, REPO)}  ({os.path.getsize(p) / 1e6:.1f} MB)")
    if a.pack_only:
        return
    lan = lan_address() if a.bind in ("0.0.0.0", "") else None
    print(f"\nServing {os.path.relpath(DIST, REPO)}/ on port {a.port}. Run on the computer that should get Jelibox:\n")
    for host in ["localhost"] + ([lan] if lan else []):
        print(f"  {'this computer' if host == 'localhost' else 'other computer'}")
        print(f"    Windows:  irm http://{host}:{a.port}/install.ps1 | iex")
        print(f"    Linux:    curl -fsSL http://{host}:{a.port}/install.sh | bash\n")
    print("Ctrl+C stops the server.")
    try:
        http.server.ThreadingHTTPServer((a.bind, a.port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
