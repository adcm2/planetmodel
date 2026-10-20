"""Fetching into the cache: served locally, by file URL and by an HTTP server
on this machine, so that nothing here touches the network."""
import hashlib
import io
import re
import sys
import tarfile
import threading
import warnings
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote

import pytest

import planetmodel
from planetmodel import datasets
from planetmodel.datasets import Dataset, Source, cache_directory, fetch, load
from planetmodel.testing import check_dataset


class Echoed(dict):
    """What `Echo` gives: what it was handed, citing the entry."""

    def cite(self):
        return "\n".join(self.citation)


class Echo:
    """A reader that gives back what it was handed."""

    def read(self, paths, dataset, **options):
        out = Echoed(paths=paths, name=dataset.name, **options)
        out.citation = dataset.citation
        return out


def sha(data):
    return hashlib.sha256(data).hexdigest()


def entry(*sources, name="m"):
    return Dataset(name, sources, Echo(), title="t", citation=("c",), licence="l")


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    """A cache of the test's own, with fetching on the network allowed and no
    proxy between this machine and its own server."""
    monkeypatch.setenv("PLANETMODEL_DATA", str(tmp_path / "cache"))
    monkeypatch.delenv("PLANETMODEL_OFFLINE", raising=False)
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.setenv(name, "*")
    return tmp_path / "cache"


@pytest.fixture
def host():
    """An HTTP server on this machine: `files` by path, a request log, a
    path under /partial/ that announces a file and sends half of it, and
    404 for anything else."""
    state = SimpleNamespace(files={}, requests=[])

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state.requests.append((self.path, self.headers.get("User-Agent")))
            path = unquote(self.path)
            cut = path.startswith("/partial/")
            data = state.files.get(path[len("/partial"):] if cut else path)
            if data is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data[:len(data) // 2] if cut else data)
            self.close_connection = True

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    yield state
    server.shutdown()
    server.server_close()


def leftovers(directory):
    """Every file in a directory, hidden partial downloads included."""
    return sorted(p.name for p in directory.rglob("*") if p.is_file())


def test_a_file_url_is_fetched_checked_and_cached(tmp_path, cache):
    data = b"S40RTS coefficients\n" * 100
    origin = tmp_path / "origin" / "m.sph"
    origin.parent.mkdir()
    origin.write_bytes(data)
    d = entry(Source(origin.as_uri(), sha(data)))
    (path,) = fetch(d)
    assert path == cache / "m" / "m.sph" and path.read_bytes() == data
    origin.unlink()                                 # the host is gone
    assert fetch(d) == (path,) and fetch(d, download=False) == (path,)
    assert leftovers(cache / "m") == ["m.sph"]


def test_over_http_once_then_from_the_cache(host, cache, monkeypatch):
    data = bytes(range(256)) * 50
    host.files["/data/S362ANI+M.nc"] = data
    d = entry(Source(f"{host.url}/data/S362ANI+M.nc", sha(data)))
    (path,) = fetch(d)
    assert path.read_bytes() == data and path.name == "S362ANI+M.nc"
    assert len(host.requests) == 1
    sent, agent = host.requests[0]
    assert sent == "/data/S362ANI%2BM.nc"           # "+" percent-encoded
    assert agent.startswith(f"planetmodel/{planetmodel.__version__}")
    fetch(d)
    monkeypatch.setenv("PLANETMODEL_OFFLINE", "1")
    fetch(d)
    assert len(host.requests) == 1                  # the cache answered


def test_offline_a_missing_file_names_where_to_place_it(host, cache, monkeypatch):
    d = entry(Source(f"{host.url}/m.sph", sha(b"x")))
    where = re.escape(str(cache / "m" / "m.sph"))
    with pytest.raises(FileNotFoundError, match=f"{where}.*{re.escape(host.url)}"):
        fetch(d, download=False)
    monkeypatch.setenv("PLANETMODEL_OFFLINE", "yes")
    with pytest.raises(FileNotFoundError, match="off the network"):
        fetch(d)
    assert host.requests == []


def test_a_wrong_hash_is_refused_naming_both_or_kept_when_asked(host, cache):
    data = b"rewritten in place"
    host.files["/m.nc"] = data
    expected = sha(b"what the entry recorded")
    d = entry(Source(f"{host.url}/m.nc", expected))
    with pytest.raises(ValueError, match=f"{sha(data)}.*{expected}"):
        fetch(d)
    assert not (cache / "m").exists() or leftovers(cache / "m") == []
    with pytest.warns(UserWarning, match=f"sha256 {sha(data)}.*kept"):
        (path,) = fetch(d, accept_changed=True)
    assert path.read_bytes() == data
    with pytest.raises(ValueError, match="cached .* remove it"):
        fetch(d)
    with pytest.warns(UserWarning, match="kept"):
        fetch(d, accept_changed=True)


def test_a_mirror_serves_when_the_first_host_fails(host, tmp_path):
    data = b"served by the mirror"
    host.files["/m.sph"] = data
    for first in (f"{host.url}/gone.sph", (tmp_path / "nowhere.sph").as_uri()):
        source = Source(first, sha(data), name="m.sph", mirrors=(f"{host.url}/m.sph",))
        d = entry(source, name=f"m{len(first)}")
        (path,) = fetch(d)
        assert path.read_bytes() == data


def test_a_mirror_with_the_entrys_file_wins_over_a_changed_one(host):
    host.files["/changed.sph"] = b"changed"
    host.files["/right.sph"] = b"right"
    d = entry(Source(f"{host.url}/changed.sph", sha(b"right"), name="m.sph",
                     mirrors=(f"{host.url}/right.sph",)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        (path,) = fetch(d, accept_changed=True)
    assert path.read_bytes() == b"right"


def test_an_interrupted_download_leaves_nothing(host, cache):
    data = b"0123456789" * 1000
    host.files["/m.sph"] = data
    d = entry(Source(f"{host.url}/partial/m.sph", sha(data), name="m.sph"))
    with pytest.raises(ConnectionError, match="stopped after 5000 of 10000 bytes"):
        fetch(d, accept_changed=True)
    assert leftovers(cache / "m") == []


def test_when_every_host_fails_the_file_can_be_placed_by_hand(host, cache):
    data = b"placed by hand"
    d = entry(Source(f"{host.url}/a.sph", sha(data), name="m.sph",
                     mirrors=(f"{host.url}/b.sph",)))
    with pytest.raises(ConnectionError, match="a.sph.*\n.*b.sph.*\n.*by hand at"):
        fetch(d)
    (cache / "m" / "m.sph").write_bytes(data)
    assert fetch(d, download=False)[0].read_bytes() == data


def tar_archive(path, files, *, links=()):
    with tarfile.open(path, "w:gz") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        for name, target in links:
            info = tarfile.TarInfo(name)
            info.type, info.linkname = tarfile.SYMTYPE, target
            tf.addfile(info)
    return path.read_bytes()


def zip_archive(path, files):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return path.read_bytes()


FILES = {"crust1.0/crust1.bnds": b"bnds", "crust1.0/sub/crust1.vp": b"vp",
         "._crust1.bnds": b"macOS", "crust1.0/readme": b"unwanted"}


@pytest.mark.parametrize("suffix, build", [(".tar.gz", tar_archive),
                                           (".zip", zip_archive)])
def test_only_the_named_members_are_extracted_once(tmp_path, cache, suffix, build):
    archive = tmp_path / f"crust1.0{suffix}"
    data = build(archive, FILES)
    members = ("crust1.0/sub/crust1.vp", "crust1.0/crust1.bnds")
    d = entry(Source(archive.as_uri(), sha(data), members=members))
    paths = fetch(d)
    into = cache / "m" / "crust1.0"
    assert paths == tuple(into / m for m in members)
    assert [p.read_bytes() for p in paths] == [b"vp", b"bnds"]
    assert leftovers(into) == [".archive-sha256", "crust1.bnds", "crust1.vp"]
    stamp = paths[0].stat().st_mtime_ns
    fetch(d)
    assert paths[0].stat().st_mtime_ns == stamp     # not extracted again
    paths[1].unlink()
    assert fetch(d)[1].read_bytes() == b"bnds"      # a lost member comes back


def test_an_archive_member_that_escapes_or_is_absent_is_refused(tmp_path):
    archive = tmp_path / "evil.tar.gz"
    data = tar_archive(archive, {"ok.txt": b"ok"},
                       links=[("up", "../../outside"), ("abs", "/etc/passwd"),
                              ("inside", "ok.txt")])
    for member, words in (("up", "refused"), ("abs", "refused"),
                          ("inside", "not a regular file"), ("gone.txt", "no member")):
        d = entry(Source(archive.as_uri(), sha(data), members=(member,)),
                  name=f"evil-{member.replace('.', '-')}")
        with pytest.raises(ValueError, match=words):
            fetch(d)
    zipped = tmp_path / "dirs.zip"
    with zipfile.ZipFile(zipped, "w") as zf:
        zf.writestr("folder/", b"")                 # a directory, named "folder/"
    d = entry(Source(zipped.as_uri(), sha(zipped.read_bytes()), members=("folder",)),
              name="dirs")
    with pytest.raises(ValueError, match="no member 'folder'"):
        fetch(d)


def test_load_hands_the_paths_and_options_to_the_reader(tmp_path):
    origin = tmp_path / "m.sph"
    origin.write_bytes(b"m")
    d = entry(Source(origin.as_uri(), sha(b"m")))
    got = load(d, lmax=12)
    assert got == {"paths": fetch(d), "name": "m", "lmax": 12}


def test_the_contract_holds_fetched_paths(tmp_path):
    archive = tmp_path / "crust1.0.tar.gz"
    data = tar_archive(archive, FILES)
    plain = tmp_path / "m.sph"
    plain.write_bytes(b"m")
    d = entry(Source(plain.as_uri(), sha(b"m")),
              Source(archive.as_uri(), sha(data), members=("crust1.0/readme",)))
    paths = fetch(d)
    check_dataset(d, paths=paths, options={"lmax": 3})
    with pytest.raises(AssertionError, match="1 paths for 2 files and members"):
        check_dataset(d, paths=paths[:1])
    with pytest.raises(AssertionError, match="sha256"):
        check_dataset(d, paths=paths[::-1])


@pytest.mark.parametrize("platform, env, want", [
    ("linux", {}, ".cache/planetmodel"),
    ("linux", {"XDG_CACHE_HOME": "relative/cache"}, ".cache/planetmodel"),
    ("darwin", {}, "Library/Caches/planetmodel"),
    ("win32", {"LOCALAPPDATA": "C:/Users/me/AppData/Local"},
     "C:/Users/me/AppData/Local/planetmodel/Cache"),
    ("win32", {}, "AppData/Local/planetmodel/Cache"),
])
def test_the_cache_is_the_platforms_unless_named(monkeypatch, platform, env, want):
    monkeypatch.delenv("PLANETMODEL_DATA")
    for name in ("XDG_CACHE_HOME", "LOCALAPPDATA"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "platform", platform)
    assert cache_directory().as_posix().endswith(want)
    monkeypatch.setenv("PLANETMODEL_DATA", "~/data")
    assert cache_directory() == Path.home() / "data"


def test_an_absolute_xdg_cache_home_is_followed(tmp_path, monkeypatch):
    monkeypatch.delenv("PLANETMODEL_DATA")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(sys, "platform", "linux")
    assert cache_directory() == tmp_path / "xdg" / "planetmodel"


def test_the_surface():
    assert set(datasets.__all__) <= set(dir(datasets))
    assert "datasets" not in planetmodel.__all__
