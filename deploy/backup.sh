#!/bin/sh
# Backups for docker-compose.production.yml. POSIX sh: it runs under busybox
# ash in the `backup` service (postgres:16-alpine) and under dash in the
# `restore-datasets` service (the application image).
#
#   backup.sh [backup]       every BACKUP_INTERVAL seconds: dump the database
#                            to BACKUP_DIR, then copy the dataset blobs it
#                            lacks from DATASET_ROOT to BACKUP_DIR/datasets
#   backup.sh restore-datasets
#                            copy the blobs DATASET_ROOT lacks back from
#                            BACKUP_DIR/datasets
#
# Order matters. A dataset is registered in the database only after its blob
# has been stored and read back, and no blob is ever deleted, so copying the
# blobs after the dump finishes puts every blob the dump refers to into the
# backup. BACKUP_DIR/datasets only grows: every dump, old or new, restores
# with it.
#
# A blob is copied under a temporary name, checked against its source's
# SHA-256 and only then renamed into place, so a copy cut short never passes
# for a blob. Only files named like blobs are copied: the research worker
# writes each blob as an unreadable (0600) .tmp-* file first, and this
# container has no capability to read it.
set -u

BACKUP_DIR=${BACKUP_DIR:-/backups}
DATASET_ROOT=${DATASET_ROOT:-/data/datasets}
BACKUP_INTERVAL=${BACKUP_INTERVAL:-21600}  # 0: one pass, exit status says whether it worked
RETENTION_DAYS=${RETENTION_DAYS:-7}
SCHEME=qs-ds-sha256-v1  # backend/services/dataset_format.py

log() {
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >&2
}

sha256() {
    set -- $(sha256sum < "$1") && echo "$1"
}

# copy_blobs SRC DST: copies each blob in SRC that DST lacks (or holds empty).
# Prints how many it copied; fails if any could not be copied.
copy_blobs() {
    src=$1
    dst=$2
    [ -d "$src" ] || { log "$src is not a directory"; return 1; }
    mkdir -p "$dst" || return 1
    # Copies left behind by a pass that was killed (never the worker's files).
    find "$dst" -type f -name '.tmp-qs-copy-*' -exec rm -f {} + || return 1
    { (cd "$src" && find . -type f) || echo "find failed"; } | {
        copied=0
        failed=0
        while IFS= read -r path; do
            case $path in
                "find failed") log "could not list $src"; failed=1; continue ;;
                ./$SCHEME/??/*) ;;
                *) continue ;;
            esac
            name=${path##*/}
            shard=${path#./$SCHEME/}
            shard=${shard%%/*}
            [ "$path" = "./$SCHEME/$shard/$name" ] || continue
            [ ${#name} -eq 64 ] || continue
            case $name in *[!0-9a-f]*) continue ;; "$shard"*) ;; *) continue ;; esac
            target=$dst/$SCHEME/$shard/$name
            [ -s "$target" ] && continue
            from=$src/$SCHEME/$shard/$name
            temporary=$dst/$SCHEME/$shard/.tmp-qs-copy-$name
            if mkdir -p "$dst/$SCHEME/$shard" && cp "$from" "$temporary" \
                && copy_sum=$(sha256 "$temporary") && from_sum=$(sha256 "$from") \
                && [ -n "$copy_sum" ] && [ "$copy_sum" = "$from_sum" ] \
                && mv -f "$temporary" "$target"; then
                copied=$((copied + 1))
            else
                rm -f "$temporary"
                log "could not copy $from"
                failed=1
            fi
        done
        echo "$copied"
        [ "$failed" -eq 0 ]
    }
}

backup_once() {
    status=0
    dumped=0
    rm -f "$BACKUP_DIR"/*.dump.partial  # left by a pass that was killed
    dump=$BACKUP_DIR/quantumsentinel-$(date +%Y%m%d%H%M%S).dump
    # Connection settings come from PGHOST, PGUSER, PGDATABASE and PGPASSWORD.
    if pg_dump -Fc > "$dump.partial" && mv "$dump.partial" "$dump"; then
        dumped=1
    else
        rm -f "$dump.partial"
        log "pg_dump failed"
        status=1
    fi
    if copied=$(copy_blobs "$DATASET_ROOT" "$BACKUP_DIR/datasets"); then
        blobs="$copied new dataset blobs"
    else
        blobs="dataset blobs incomplete (${copied:-0} copied)"
        status=1
    fi
    # Old dumps go only once a new one exists; the dataset backup is never pruned.
    if [ "$dumped" -eq 1 ]; then
        find "$BACKUP_DIR" -maxdepth 1 -type f -name 'quantumsentinel-*.dump' -mtime +"$RETENTION_DAYS" \
            -exec rm -f {} + || status=1
    fi
    if [ "$status" -eq 0 ]; then
        log "backup ok: ${dump##*/}, $blobs"
    else
        log "backup failed: $blobs"
    fi
    return "$status"
}

case ${1:-backup} in
    backup)
        if [ "$BACKUP_INTERVAL" -eq 0 ]; then
            backup_once
            exit
        fi
        trap 'exit 143' TERM INT
        while :; do
            backup_once || :
            sleep "$BACKUP_INTERVAL" &
            wait $!
        done
        ;;
    restore-datasets)
        if copied=$(copy_blobs "$BACKUP_DIR/datasets" "$DATASET_ROOT"); then
            log "restore ok: $copied dataset blobs copied to $DATASET_ROOT"
        else
            log "restore failed: ${copied:-0} dataset blobs copied to $DATASET_ROOT"
            exit 1
        fi
        ;;
    *)
        log "usage: backup.sh [backup | restore-datasets]"
        exit 2
        ;;
esac
