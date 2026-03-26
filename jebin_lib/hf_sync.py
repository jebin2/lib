# HF Sync
#
# Two directions:
#
# 1. local CONTENT_TO_BE_PROCESSED  →  HF_MOUNT_PATH   (sync_to_hf)
#    All processing writes locally; sync_to_hf pushes to the NFS mount.
#
#      a. Before each pipeline run  →  sync_to_hf(src, dst, subpath=...)
#         - Skips if called within SYNC_INTERVAL (5 min) of the last sync.
#         - Copy pass : walks src, copies new/modified files to dst.
#         - Delete pass: walks dst, removes anything no longer present in src.
#         - Updates _last_sync_time after completion.
#
#      b. After publisher _cleanup_folder()  →  sync_to_hf(src, dst, subpath=..., force=True)
#         - Bypasses the cooldown and runs immediately.
#         - Same copy + delete pass so deleted files are removed from HF right away.
#         - Does NOT update _last_sync_time so the next scheduled sync still runs on time.
#
#      subpath="category/video_name": scopes the sync to that subfolder only;
#         parent dirs in dst are created automatically.
#
# 2. HF_BUCKET_ID  →  local CONTENT_TO_BE_PROCESSED    (sync_from_hf)
#    Downloads the full HF bucket content directly via the HF API (no NFS mount needed).
#    Triggered by passing --syncfromhf on the command line; runs once then exits.

import os
import shutil
import time

from custom_logger import logger_config

SYNC_INTERVAL = 300  # 5 minutes

_last_sync_time = 0


def _delete_pass(src, dst):
    """Remove files/dirs from dst that no longer exist in src."""
    for root, dirs, files in os.walk(dst, topdown=False):
        rel_root = os.path.relpath(root, dst)
        src_dir = os.path.join(src, rel_root) if rel_root != '.' else src
        for fname in files:
            if not os.path.exists(os.path.join(src_dir, fname)):
                dst_file = os.path.join(root, fname)
                try:
                    os.remove(dst_file)
                    logger_config.info(f"HF sync: removed {dst_file}")
                except Exception as e:
                    logger_config.warning(f"HF sync: failed to remove {dst_file}: {e}")
        if not os.path.exists(src_dir) and root != dst:
            try:
                os.rmdir(root)
                logger_config.info(f"HF sync: removed dir {root}")
            except OSError:
                pass  # not empty yet, cleaned on next iteration


def sync_to_hf(src_base, dst_base, subpath=None, force=False):
    """Sync src_base to dst_base.

    subpath: optional relative path under src_base/dst_base to sync instead of the full tree.
             Parent dirs in dst_base are created automatically.
    force=False (default): copy + delete pass, subject to SYNC_INTERVAL cooldown.
    force=True: copy + delete pass immediately, bypassing the cooldown.
    """
    global _last_sync_time
    if not dst_base:
        return

    src = os.path.join(src_base, subpath) if subpath else src_base
    dst = os.path.join(dst_base, subpath) if subpath else dst_base

    if not force and (time.time() - _last_sync_time) < SYNC_INTERVAL:
        return

    try:
        logger_config.info(f"HF sync: {src} -> {dst}")
        os.makedirs(dst, exist_ok=True)

        # --- copy new/modified files ---
        for root, dirs, files in os.walk(src):
            rel_root = os.path.relpath(root, src)
            dst_dir = os.path.join(dst, rel_root) if rel_root != '.' else dst
            os.makedirs(dst_dir, exist_ok=True)
            for fname in files:
                src_file = os.path.join(root, fname)
                dst_file = os.path.join(dst_dir, fname)
                try:
                    if not os.path.exists(dst_file) or os.path.getmtime(src_file) > os.path.getmtime(dst_file):
                        shutil.copy2(src_file, dst_file)
                except OSError as e:
                    if e.errno == 116:  # ESTALE — src is stale, remove from dst
                        logger_config.warning(f"HF sync: stale file handle on {src_file}, removing from HF mount")
                        try:
                            os.remove(dst_file)
                        except Exception:
                            pass
                    else:
                        logger_config.warning(f"HF sync: failed to copy {src_file}: {e}")
                except Exception as e:
                    logger_config.warning(f"HF sync: failed to copy {src_file}: {e}")

        _delete_pass(src, dst)
        if not force:
            _last_sync_time = time.time()
        logger_config.info("HF sync completed")
    except Exception as e:
        logger_config.error(f"HF sync failed: {e}")


def sync_from_hf(dst, bucket_id, hf_token):
    """Download the full HF bucket directly to dst via the HF API (no NFS mount needed)."""
    if not bucket_id or not hf_token:
        logger_config.warning("sync_from_hf: HF_BUCKET_ID or HF_TOKEN not set, skipping.")
        return
    try:
        from .hf_bucket_client import HFBucketClient
        logger_config.info(f"Syncing from HF bucket {bucket_id} -> {dst}")
        HFBucketClient(token=hf_token, bucket_id=bucket_id).download_folder(remote_path="", local_folder=dst)
        logger_config.info("sync_from_hf completed")
    except Exception as e:
        logger_config.error(f"sync_from_hf failed: {e}")
