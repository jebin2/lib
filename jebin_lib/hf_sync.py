# HF Sync — local CONTENT_TO_BE_PROCESSED  →  HF_MOUNT_PATH
#
# Flow:
#   1. All processing writes to local CONTENT_TO_BE_PROCESSED (never directly to HF).
#   2. sync_to_hf() is called explicitly at two points:
#
#      a. End of each main loop iteration  →  sync_to_hf(src, dst)
#         - Skips if called within SYNC_INTERVAL (5 min) of the last sync.
#         - Copy pass : walks src, copies new/modified files to dst.
#         - Delete pass: walks dst, removes anything no longer present in src.
#         - Updates _last_sync_time after completion.
#
#      b. After publisher _cleanup_folder()  →  sync_to_hf(src, dst, force=True)
#         - Bypasses the cooldown and runs immediately.
#         - Same copy + delete pass so deleted files are removed from HF right away.
#         - Does NOT update _last_sync_time so the next scheduled sync still runs on time.
#
#   3. Subfolder sync  →  sync_to_hf(src, dst, subpath="category/video_name")
#         - Scopes the copy + delete to src/subpath → dst/subpath only.
#         - Parent dirs in dst are created automatically.
#         - Works with force=True as well.

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
                except Exception as e:
                    logger_config.warning(f"HF sync: failed to copy {src_file}: {e}")

        _delete_pass(src, dst)
        if not force:
            _last_sync_time = time.time()
        logger_config.info("HF sync completed")
    except Exception as e:
        logger_config.error(f"HF sync failed: {e}")
