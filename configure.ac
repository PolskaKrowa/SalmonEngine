# -------------------------------------------------------------
#  configure.ac — Salmon Chess Engine
#
#  Run:  autoreconf --install
#  Then: ./configure [--prefix=/usr/local]
#                    [--enable-debug]
#                    [--enable-sanitize]
#                    [--enable-tuner]
#                    [--enable-dist-tuner]
#
#  --enable-tuner      builds bin/tuna  (single-node NNUE tuner)
#  --enable-dist-tuner builds bin/tunad (distributed MPI tuner)
#                      implies --enable-tuner (shares object files)
# -------------------------------------------------------------

AC_INIT([salmon], [2.1], [], [salmon-engine])
AC_CONFIG_MACRO_DIRS([m4])
AC_CONFIG_SRCDIR([src/main.c])          # sanity-check: fails if this file is missing
AC_CONFIG_HEADERS([include/config.h])   # generates a config.h with detected defines
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([foreign -Wall -Werror subdir-objects])

# ── C compiler ────────────────────────────────────────────────────────────

AC_PROG_CC

# Inline check for -std=c2x (replaces AX_CHECK_COMPILE_FLAG from autoconf-archive)
AC_MSG_CHECKING([whether $CC accepts -std=c2x])
save_CFLAGS="$CFLAGS"
CFLAGS="$CFLAGS -std=c2x"
AC_COMPILE_IFELSE([AC_LANG_PROGRAM()],
    [AC_MSG_RESULT([yes]); CFLAGS="$save_CFLAGS -std=c2x"],
    [AC_MSG_RESULT([no])
     CFLAGS="$save_CFLAGS"
     AC_MSG_ERROR([Your compiler does not support -std=c2x. GCC 9+ or Clang 9+ required.])])

# Inline check for -march=native
AC_MSG_CHECKING([whether $CC accepts -march=native])
save_CFLAGS="$CFLAGS"
CFLAGS="$CFLAGS -march=native"
AC_COMPILE_IFELSE([AC_LANG_PROGRAM()],
    [AC_MSG_RESULT([yes]); MARCH_NATIVE="-march=native"],
    [AC_MSG_RESULT([no])
     AC_MSG_WARN([-march=native not supported; CPU-specific optimisations disabled.])
     MARCH_NATIVE=""])
CFLAGS="$save_CFLAGS"
AC_SUBST([MARCH_NATIVE])

# Detect CPU SIMD support for NNUE optimisations
AC_MSG_CHECKING([whether $CC accepts -mavx2])
save_CFLAGS="$CFLAGS"
CFLAGS="$CFLAGS -mavx2"
AC_COMPILE_IFELSE([AC_LANG_PROGRAM()],
    [AC_MSG_RESULT([yes]); HAVE_AVX2=1],
    [AC_MSG_RESULT([no]); HAVE_AVX2=0])
CFLAGS="$save_CFLAGS"
AC_DEFINE_UNQUOTED([HAVE_AVX2], [$HAVE_AVX2], [Define to 1 if AVX2 is available])
AC_SUBST([HAVE_AVX2])

AC_MSG_CHECKING([whether $CC accepts -mbmi2])
save_CFLAGS="$CFLAGS"
CFLAGS="$CFLAGS -mbmi2"
AC_COMPILE_IFELSE([AC_LANG_PROGRAM()],
    [AC_MSG_RESULT([yes]); HAVE_BMI2=1],
    [AC_MSG_RESULT([no]); HAVE_BMI2=0])
CFLAGS="$save_CFLAGS"
AC_DEFINE_UNQUOTED([HAVE_BMI2], [$HAVE_BMI2], [Define to 1 if BMI2 is available])
AC_SUBST([HAVE_BMI2])

# ── Required builtins ─────────────────────────────────────────────────────

AC_MSG_CHECKING([for __builtin_popcountll])
AC_LINK_IFELSE(
    [AC_LANG_PROGRAM([[]], [[long long x = __builtin_popcountll(0ULL);]])],
    [AC_MSG_RESULT([yes])],
    [AC_MSG_ERROR([__builtin_popcountll not available. Use GCC or Clang.])])

AC_MSG_CHECKING([for __builtin_ctzll])
AC_LINK_IFELSE(
    [AC_LANG_PROGRAM([[]], [[long long x = __builtin_ctzll(1ULL);]])],
    [AC_MSG_RESULT([yes])],
    [AC_MSG_ERROR([__builtin_ctzll not available. Use GCC or Clang.])])

# ── Always-required libraries ─────────────────────────────────────────────

AC_CHECK_LIB([m], [sqrt], [],
    [AC_MSG_ERROR([libm not found. Install libc development headers.])])

AC_CHECK_LIB([pthread], [pthread_create], [],
    [AC_MSG_ERROR([libpthread not found. Install libc development headers.])])

# ── Build-mode flags ──────────────────────────────────────────────────────

AC_ARG_ENABLE([debug],
    [AS_HELP_STRING([--enable-debug], [build with debug symbols and no optimisation])],
    [enable_debug=$enableval], [enable_debug=no])

AC_ARG_ENABLE([sanitize],
    [AS_HELP_STRING([--enable-sanitize], [build with AddressSanitizer + UBSan])],
    [enable_sanitize=$enableval], [enable_sanitize=no])

if test "x$enable_sanitize" = "xyes"; then
    CFLAGS="$CFLAGS -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer"
    LDFLAGS="$LDFLAGS -fsanitize=address,undefined"
    AC_MSG_NOTICE([Sanitize build enabled (ASan + UBSan)])
elif test "x$enable_debug" = "xyes"; then
    CFLAGS="$CFLAGS -O0 -g3 $MARCH_NATIVE -DENGINE_DEBUG"
    AC_MSG_NOTICE([Debug build enabled])
else
    CFLAGS="$CFLAGS -O3 $MARCH_NATIVE -DNDEBUG -fomit-frame-pointer -flto"
    LDFLAGS="$LDFLAGS -flto"
    AC_MSG_NOTICE([Release build enabled])
fi

# ── Optional: single-node tuner (bin/tuna) ────────────────────────────────

AC_ARG_ENABLE([tuner],
    [AS_HELP_STRING([--enable-tuner], [build the single-node tuner binary (bin/tuna)])],
    [enable_tuner=$enableval], [enable_tuner=no])

# ── Optional: distributed tuner (bin/tunad) ───────────────────────────────
#
# --enable-dist-tuner implies --enable-tuner because tunad links the same
# engine object files and tune.c (compiled without -DTUNE_STANDALONE).

AC_ARG_ENABLE([dist-tuner],
    [AS_HELP_STRING([--enable-dist-tuner],
        [build the distributed MPI tuner (bin/tunad); implies --enable-tuner])],
    [enable_dist_tuner=$enableval], [enable_dist_tuner=no])

if test "x$enable_dist_tuner" = "xyes"; then
    enable_tuner=yes
fi

# ── OpenBLAS (required by both tuner variants) ────────────────────────────
#
# Kept as a per-target LDADD variable (OPENBLAS_LIBS) instead of appending
# to the global LDFLAGS, so the engine binary is not linked against OpenBLAS.

OPENBLAS_LIBS=""
if test "x$enable_tuner" = "xyes"; then
    AC_CHECK_LIB([openblas], [cblas_sgemm],
        [OPENBLAS_LIBS="-lopenblas"],
        [AC_MSG_ERROR([OpenBLAS not found. Install libopenblas-dev.])])
fi
AC_SUBST([OPENBLAS_LIBS])

# ── OpenSSL (required by dist tuner only) — via pkg-config ───────────────

OPENSSL_CFLAGS=""
OPENSSL_LIBS=""
if test "x$enable_dist_tuner" = "xyes"; then
    PKG_CHECK_MODULES([OPENSSL], [openssl >= 3.0],
        [],
        [AC_MSG_ERROR([OpenSSL >= 3.0 not found. Install libssl-dev.])])
fi
AC_SUBST([OPENSSL_CFLAGS])
AC_SUBST([OPENSSL_LIBS])

# ── OpenMPI C bindings (required by dist tuner only) — via pkg-config ─────
#
# ompi-c.pc is the canonical C-only pkg-config module installed by OpenMPI.
# Using pkg-config here rather than AC_CHECK_LIB means we automatically pick
# up non-standard install prefixes (e.g. /opt/openmpi) without the user
# having to manually set CPPFLAGS/LDFLAGS.

MPI_CFLAGS=""
MPI_LIBS=""
if test "x$enable_dist_tuner" = "xyes"; then
    PKG_CHECK_MODULES([MPI], [ompi-c >= 4.0],
        [],
        [AC_MSG_ERROR([OpenMPI C >= 4.0 not found. Install libopenmpi-dev.])])

    # Verify MPI_Init is actually linkable with the flags pkg-config gave us.
    # This catches installations where the .pc file is present but the .so
    # was built for a different ABI.
    save_CFLAGS="$CFLAGS"
    save_LIBS="$LIBS"
    CFLAGS="$CFLAGS $MPI_CFLAGS"
    LIBS="$LIBS $MPI_LIBS"
    AC_CHECK_FUNC([MPI_Init], [],
        [AC_MSG_ERROR([MPI_Init not found with flags from ompi-c.pc.])])
    CFLAGS="$save_CFLAGS"
    LIBS="$save_LIBS"
fi
AC_SUBST([MPI_CFLAGS])
AC_SUBST([MPI_LIBS])

# ── Automake conditionals ─────────────────────────────────────────────────

AM_CONDITIONAL([ENABLE_TUNER],      [test "x$enable_tuner"      = "xyes"])
AM_CONDITIONAL([ENABLE_DIST_TUNER], [test "x$enable_dist_tuner" = "xyes"])

# ── Output ────────────────────────────────────────────────────────────────

AC_CONFIG_FILES([Makefile src/Makefile])
AC_OUTPUT

AC_MSG_NOTICE([
-------------------------------------------
  Salmon Chess Engine — configuration done
-------------------------------------------
  Build targets:
    engine  (always)
    tuna    (--enable-tuner or --enable-dist-tuner)
    tunad   (--enable-dist-tuner)

  Now run:
    make
    make check   (perft smoke test)

  Distributed tuner quick-start:
    export TUNE_HMAC_KEY_HEX=\$(openssl rand -hex 32)
    mpirun -np 4 ./src/tunad --threads 4 --games 2000
])
