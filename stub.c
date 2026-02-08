/* stub.c — pypack native launcher
 *
 * Compile:
 *   Linux:  cc -O2 -s -o stub stub.c
 *   macOS:  cc -O2 -o stub stub.c
 *
 * No library deps beyond libc. Runtime tools needed on target:
 *   zstd, tar, sha256sum (Linux) or shasum (macOS)
 *
 * At runtime it:
 *  1. Finds the trailer (last 32 bytes of itself)
 *  2. Hashes the runtime blob → cache key
 *  3. Extracts runtime to ~/.cache/pypack/<hash>/ if not cached
 *  4. Execs: <cached python3> <self> [original args...]
 *     Python's zipimport reads __main__.py from the zip at the tail of <self>
 */

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

/* ── Trailer format (last 32 bytes of the packed binary) ──────────── */

#define MAGIC       "PYPK\x00\x01\x00\x00"
#define MAGIC_LEN   8
#define TRAILER_SZ  32

typedef struct {
    char     magic[8];
    uint64_t runtime_offset;
    uint64_t runtime_size;
    uint64_t app_offset;
} __attribute__((packed)) Trailer;

/* ── Get path to own executable ───────────────────────────────────── */

static int self_exe(char *buf, size_t n) {
#if defined(__linux__)
    ssize_t len = readlink("/proc/self/exe", buf, n - 1);
    if (len < 0) return -1;
    buf[len] = '\0';
    return 0;
#elif defined(__APPLE__)
    uint32_t sz = (uint32_t)n;
    if (_NSGetExecutablePath(buf, &sz) != 0) return -1;
    char real[PATH_MAX];
    if (realpath(buf, real)) strncpy(buf, real, n);
    return 0;
#else
    #error "Unsupported platform — see roadmap for Windows support"
#endif
}

/* ── Hash a file region → first 16 hex chars of SHA-256 ───────────── */

static int hash_region(const char *path, uint64_t off, uint64_t sz,
                       char *hex, size_t hexsz) {
    char cmd[PATH_MAX + 256];
#if defined(__APPLE__)
    /* macOS dd lacks iflag=skip_bytes; use tail+head instead */
    snprintf(cmd, sizeof(cmd),
        "tail -c +%llu '%s' | head -c %llu | shasum -a 256 | cut -c1-16",
        (unsigned long long)(off + 1), path, (unsigned long long)sz);
#else
    snprintf(cmd, sizeof(cmd),
        "dd if='%s' bs=4096 skip=%llu count=%llu iflag=skip_bytes,count_bytes "
        "2>/dev/null | sha256sum | cut -c1-16",
        path, (unsigned long long)off, (unsigned long long)sz);
#endif

    FILE *fp = popen(cmd, "r");
    if (!fp) return -1;
    if (!fgets(hex, hexsz, fp)) { pclose(fp); return -1; }
    pclose(fp);
    char *nl = strchr(hex, '\n');
    if (nl) *nl = '\0';
    return 0;
}

/* ── Extract: dd | zstd -d | tar xf ──────────────────────────────── */

static int extract(const char *path, uint64_t off, uint64_t sz,
                   const char *dest) {
    char cmd[PATH_MAX * 2 + 512];
#if defined(__APPLE__)
    snprintf(cmd, sizeof(cmd),
        "tail -c +%llu '%s' | head -c %llu | zstd -d 2>/dev/null | tar xf - -C '%s' 2>/dev/null",
        (unsigned long long)(off + 1), path, (unsigned long long)sz, dest);
#else
    snprintf(cmd, sizeof(cmd),
        "dd if='%s' bs=4096 skip=%llu count=%llu iflag=skip_bytes,count_bytes "
        "2>/dev/null | zstd -d 2>/dev/null | tar xf - -C '%s' 2>/dev/null",
        path, (unsigned long long)off, (unsigned long long)sz, dest);
#endif
    int r = system(cmd);
    return WIFEXITED(r) ? WEXITSTATUS(r) : -1;
}

/* ── mkdir -p ─────────────────────────────────────────────────────── */

static void mkdirp(const char *path) {
    char tmp[PATH_MAX];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') { *p = 0; mkdir(tmp, 0755); *p = '/'; }
    }
    mkdir(tmp, 0755);
}

/* ── Locate python3 in the extracted tree ─────────────────────────── */

static int find_python(const char *dir, char *out, size_t n) {
    /* python-build-standalone ships several possible layouts depending
     * on the variant (install_only, install_only_stripped, full) and
     * the version. We try the known paths in order. */
    const char *tries[] = {
        "%s/python/install/bin/python3",
        "%s/install/bin/python3",
        "%s/bin/python3",
        "%s/python/bin/python3",
        NULL
    };
    for (const char **t = tries; *t; t++) {
        snprintf(out, n, *t, dir);
        if (access(out, X_OK) == 0) return 0;
    }
    return -1;
}

/* ── Main ─────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    char me[PATH_MAX];
    if (self_exe(me, sizeof(me)) != 0) {
        fprintf(stderr, "pypack: can't determine own executable path\n");
        return 1;
    }

    /* ── Read trailer (last 32 bytes) ── */
    int fd = open(me, O_RDONLY);
    if (fd < 0) { perror("pypack: open self"); return 1; }

    off_t fsz = lseek(fd, 0, SEEK_END);
    if (fsz < TRAILER_SZ) {
        fprintf(stderr, "pypack: binary too small to contain trailer\n");
        close(fd);
        return 1;
    }

    Trailer t;
    lseek(fd, fsz - TRAILER_SZ, SEEK_SET);
    if (read(fd, &t, TRAILER_SZ) != TRAILER_SZ) {
        fprintf(stderr, "pypack: failed to read trailer\n");
        close(fd);
        return 1;
    }
    close(fd);

    /* ── Validate magic ── */
    if (memcmp(t.magic, MAGIC, MAGIC_LEN) != 0) {
        fprintf(stderr, "pypack: invalid magic (corrupted or not a pypack binary)\n");
        return 1;
    }

    /* ── Compute cache directory ── */
    char hash[32];
    if (hash_region(me, t.runtime_offset, t.runtime_size,
                    hash, sizeof(hash)) != 0) {
        fprintf(stderr, "pypack: failed to hash runtime blob\n");
        return 1;
    }

    const char *home = getenv("HOME");
    if (!home) home = "/tmp";

    /* Respect XDG_CACHE_HOME if set */
    const char *xdg_cache = getenv("XDG_CACHE_HOME");
    char cache[PATH_MAX], py[PATH_MAX];
    if (xdg_cache && xdg_cache[0]) {
        snprintf(cache, sizeof(cache), "%s/pypack/%s", xdg_cache, hash);
    } else {
        snprintf(cache, sizeof(cache), "%s/.cache/pypack/%s", home, hash);
    }

    /* ── Extract runtime if not cached ── */
    if (find_python(cache, py, sizeof(py)) != 0) {
        fprintf(stderr, "pypack: first run — extracting Python runtime...\n");
        mkdirp(cache);

        if (extract(me, t.runtime_offset, t.runtime_size, cache) != 0) {
            fprintf(stderr, "pypack: runtime extraction failed\n");
            fprintf(stderr, "        ensure 'zstd' and 'tar' are installed\n");
            return 1;
        }

        if (find_python(cache, py, sizeof(py)) != 0) {
            fprintf(stderr, "pypack: can't find python3 in extracted runtime at %s\n", cache);
            return 1;
        }

        fprintf(stderr, "pypack: runtime cached at %s\n", cache);
    }

    /* ── Build argv for exec ──
     *
     *   python3 <self_path> [user_arg1] [user_arg2] ...
     *
     * When Python receives a zip file (or a file with a zip appended) as
     * the script argument, it uses zipimport to find __main__.py inside
     * the zip and executes it. The app zip is at the tail of <self_path>,
     * so this "just works".
     */
    int new_argc = argc + 1;
    char **new_argv = malloc(sizeof(char *) * (new_argc + 1));
    if (!new_argv) {
        perror("pypack: malloc");
        return 1;
    }

    new_argv[0] = py;       /* python3 */
    new_argv[1] = me;       /* the packed binary (= zip with app) */
    for (int i = 1; i < argc; i++) {
        new_argv[i + 1] = argv[i];  /* pass through user args */
    }
    new_argv[new_argc] = NULL;

    /* ── Set environment ── */
    setenv("_PYPACK_SELF", me, 1);       /* so bootstrap knows the binary path */
    setenv("_PYPACK_CACHE", cache, 1);   /* so bootstrap can find native libs (v2) */
    setenv("PYTHONNOUSERSITE", "1", 1);  /* isolate from user site-packages */

    /* ── Replace this process with Python ── */
    execv(py, new_argv);

    /* If we reach here, exec failed */
    perror("pypack: exec python3");
    return 1;
}
