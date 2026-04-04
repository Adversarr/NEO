"""A package for downloading and processing Objaverse."""

import glob
import gzip
import json
import multiprocessing
import os
import warnings
from typing import Any

from tqdm import tqdm
import requests

OBJAVERSE_OVERRIDE_PATH = os.getenv("OBJAVERSE_OVERRIDE_PATH", None)
if OBJAVERSE_OVERRIDE_PATH is not None:
    BASE_PATH = OBJAVERSE_OVERRIDE_PATH
    print(f"Using OBJAVERSE_OVERRIDE_PATH: {BASE_PATH}")
else:
    BASE_PATH = os.path.join(os.path.expanduser("~"), ".objaverse")
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

__version__ = "0.1.7"
_VERSIONED_PATH = os.path.join(BASE_PATH, "hf-objaverse-v1")

def _download_with_requests(url: str, local_path: str, timeout: int = 20, max_retries: int = 3) -> None:
    """Download a file using requests with timeout and retries.

    Args:
        url: URL to download.
        local_path: Local path to save the file.
        timeout: Total timeout in seconds.
        max_retries: Maximum number of retry attempts.
    """
    import time
    
    proxy_config = {
        'https': 'http://127.0.0.1:7890',
        'http': 'http://127.0.0.1:7890'
    }
    
    for attempt in range(max_retries):
        try:
            # Timeout configuration: connect=10s, read=max(1, timeout-10)s
            response = requests.get(
                url,
                timeout=(5, max(1, timeout - 5)),
                stream=True,
                proxies=proxy_config if attempt > 0 else None
            )
            response.raise_for_status()  # Check HTTP status
            
            # Write file
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            # Download succeeded
            return
            
        except requests.exceptions.Timeout:
            print(f"Download timed out: {url} (>{timeout}s), attempt {attempt + 1}/{max_retries}")
            if os.path.exists(local_path):
                os.remove(local_path)
            
            if attempt < max_retries - 1:
                wait_time = 2  ** attempt  # Exponential backoff: 1, 2, 4 seconds
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Reached max retries ({max_retries}), download failed: {url}")
                raise
                
        except requests.exceptions.RequestException as e:
            print(f"Download failed: {url}, error: {e}, attempt {attempt + 1}/{max_retries}")
            if os.path.exists(local_path):
                os.remove(local_path)
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Reached max retries ({max_retries}), download failed: {url}")
                raise

def load_annotations(uids: list[str] | None = None) -> dict[str, Any]:
    """Load the full metadata of all objects in the dataset.

    Args:
        uids: A list of uids with which to load metadata. If None, it loads
        the metadata for all uids.

    Returns:
        A dictionary mapping the uid to the metadata.
    """
    metadata_path = os.path.join(_VERSIONED_PATH, "metadata")
    object_paths = _load_object_paths()
    dir_ids = (
        set(object_paths[uid].split("/")[1] for uid in uids)
        if uids is not None
        else [f"{i // 1000:03d}-{i % 1000:03d}" for i in range(160)]
    )
    if len(dir_ids) > 10:
        dir_ids = tqdm(dir_ids)
    out = {}
    for i_id in dir_ids:
        json_file = f"{i_id}.json.gz"
        local_path = os.path.join(metadata_path, json_file)
        if not os.path.exists(local_path):
            hf_url = f"{HF_ENDPOINT}/datasets/allenai/objaverse/resolve/main/metadata/{i_id}.json.gz"
            # wget the file and put it in local_path
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            _download_with_requests(hf_url, local_path)
        with gzip.open(local_path, "rb") as f:
            data = json.load(f)
        if uids is not None:
            data = {uid: data[uid] for uid in uids if uid in data}
        out.update(data)
        if uids is not None and len(out) == len(uids):
            break
    return out


def _load_object_paths() -> dict[str, str]:
    """Load the object paths from the dataset.

    The object paths specify the location of where the object is located
    in the Hugging Face repo.

    Returns:
        A dictionary mapping the uid to the object path.
    """
    object_paths_file = "object-paths.json.gz"
    local_path = os.path.join(_VERSIONED_PATH, object_paths_file)
    if not os.path.exists(local_path):
        hf_url = f"{HF_ENDPOINT}/datasets/allenai/objaverse/resolve/main/{object_paths_file}"
        # wget the file and put it in local_path
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        _download_with_requests(hf_url, local_path)
    with gzip.open(local_path, "rb") as f:
        object_paths = json.load(f)
    return object_paths


def load_uids() -> list[str]:
    """Load the uids from the dataset.

    Returns:
        A list of uids.
    """
    return list(_load_object_paths().keys())


def _download_object(
    uid: str,
    object_path: str,
    total_downloads: float,
    start_file_count: int,
) -> tuple[str, str]:
    """Download the object for the given uid.

    Args:
        uid: The uid of the object to load.
        object_path: The path to the object in the Hugging Face repo.

    Returns:
        The local path of where the object was downloaded.
    """
    # print(f"downloading {uid}")
    local_path = os.path.join(_VERSIONED_PATH, object_path)
    tmp_local_path = os.path.join(_VERSIONED_PATH, object_path + ".tmp")
    hf_url = f"{HF_ENDPOINT}/datasets/allenai/objaverse/resolve/main/{object_path}"
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    try:
        _download_with_requests(hf_url, tmp_local_path)
    except Exception as e:
        print(f"Download failed: {uid}, error: {e}")
        if os.path.exists(tmp_local_path):
            os.remove(tmp_local_path)
        return "", ""

    try:
        os.rename(tmp_local_path, local_path)
    except Exception as e:
        print(f"Rename failed: {uid}, error: {e}")
        if os.path.exists(tmp_local_path):
            os.remove(tmp_local_path)
        return "", ""

    files = glob.glob(os.path.join(_VERSIONED_PATH, "glbs", "*", "*.glb"))
    print(
        "Downloaded",
        len(files) - start_file_count,
        "/",
        total_downloads,
        "objects",
    )

    return uid, local_path


def load_objects(uids: list[str], download_processes: int = 1) -> dict[str, str]:
    """Return the path to the object files for the given uids.

    If the object is not already downloaded, it will be downloaded.

    Args:
        uids: A list of uids.
        download_processes: The number of processes to use to download the objects.

    Returns:
        A dictionary mapping the object uid to the local path of where the object
        downloaded.
    """
    object_paths = _load_object_paths()
    out = {}
    if download_processes == 1:
        uids_to_download = []
        for uid in uids:
            if uid.endswith(".glb"):
                uid = uid[:-4]
            if uid not in object_paths:
                warnings.warn(f"Could not find object with uid {uid}. Skipping it.")
                continue
            object_path = object_paths[uid]
            local_path = os.path.join(_VERSIONED_PATH, object_path)
            if os.path.exists(local_path):
                out[uid] = local_path
                continue
            uids_to_download.append((uid, object_path))
        if len(uids_to_download) == 0:
            return out
        start_file_count = len(glob.glob(os.path.join(_VERSIONED_PATH, "glbs", "*", "*.glb")))
        for uid, object_path in uids_to_download:
            uid, local_path = _download_object(uid, object_path, len(uids_to_download), start_file_count)
            out[uid] = local_path
    else:
        args = []
        for uid in uids:
            if uid.endswith(".glb"):
                uid = uid[:-4]
            if uid not in object_paths:
                warnings.warn(f"Could not find object with uid {uid}. Skipping it.")
                continue
            object_path = object_paths[uid]
            local_path = os.path.join(_VERSIONED_PATH, object_path)
            if not os.path.exists(local_path):
                args.append((uid, object_paths[uid]))
            else:
                out[uid] = local_path
        if len(args) == 0:
            return out
        print(f"starting download of {len(args)} objects with {download_processes} processes")
        start_file_count = len(glob.glob(os.path.join(_VERSIONED_PATH, "glbs", "*", "*.glb")))
        args_list = [(*arg, len(args), start_file_count) for arg in args]
        with multiprocessing.Pool(download_processes) as pool:
            r = pool.starmap(_download_object, args_list)
            for uid, local_path in r:
                out[uid] = local_path

    out = {k: v for k, v in out.items() if v != ""}
    return out


def load_lvis_annotations() -> dict[str, list[str]]:
    """Load the LVIS annotations.

    If the annotations are not already downloaded, they will be downloaded.

    Returns:
        A dictionary mapping the LVIS category to the list of uids in that category.
    """
    hf_url = f"{HF_ENDPOINT}/datasets/allenai/objaverse/resolve/main/lvis-annotations.json.gz"
    local_path = os.path.join(_VERSIONED_PATH, "lvis-annotations.json.gz")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        _download_with_requests(hf_url, local_path)
    with gzip.open(local_path, "rb") as f:
        lvis_annotations = json.load(f)
    return lvis_annotations
