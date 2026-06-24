# -------------------------------------------------------------
#  configure.ac — Salmon Chess Engine
#
#  Run:  autoreconf --install
#  Then: ./configure [--prefix=/usr/local] [--enable-debug]
#                    [--enable-sanitize] [--enable-tuner]
#                    [--enable-eval-debug]
#                    [--enable-superopt]
#                    [--with-clang=PATH] [--with-opt=PATH] [--with-llc=PATH]
#                    [--with-souper-pass=PATH] [--with-minotaur-pass=PATH]
#                    [--with-souper=PATH]
# -------------------------------------------------------------

AC_INIT([salmon], [2.0], [], [salmon-engine])
AC_CONFIG_MACRO_DIRS([m4])
AC_CONFIG_SRCDIR([src/main.c])          # sanity-check: fails if this file is missing
AC_CONFIG_HEADERS([include/config.h])   # generates a config.h with detected defines
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([foreign -Wall -Werror subdir-objects])

# Require a C compiler that supports C2x
AC_PROG_CC
AX_CHECK_COMPILE_FLAG([-std=c2x],
    [CFLAGS="$CFLAGS -std=c2x"],
    [AC_MSG_ERROR([Your compiler does not support -std=c2x. GCC 9+ or Clang 9+ required.])])

# Check for -march=native support
AX_CHECK_COMPILE_FLAG([-march=native],
    [MARCH_NATIVE="-march=native"],
    [AC_MSG_WARN([-march=native not supported; CPU-specific optimisations disabled.])
     MARCH_NATIVE=""])
AC_SUBST([MARCH_NATIVE])

# Check for __builtin_popcountll (needed by bitboard code)
AC_MSG_CHECKING([for __builtin_popcountll])
AC_LINK_IFELSE(
    [AC_LANG_PROGRAM([[]], [[long long x = __builtin_popcountll(0ULL);]])],
    [AC_MSG_RESULT([yes])],
    [AC_MSG_ERROR([__builtin_popcountll not available. Use GCC or Clang.])])

# Check for __builtin_ctzll
AC_MSG_CHECKING([for __builtin_ctzll])
AC_LINK_IFELSE(
    [AC_LANG_PROGRAM([[]], [[long long x = __builtin_ctzll(1ULL);]])],
    [AC_MSG_RESULT([yes])],
    [AC_MSG_ERROR([__builtin_ctzll not available. Use GCC or Clang.])])

# libm
AC_CHECK_LIB([m], [sqrt], [],
    [AC_MSG_ERROR([libm not found. Install libc development headers.])])

# Optional build modes
AC_ARG_ENABLE([debug],
    [AS_HELP_STRING([--enable-debug], [build with debug symbols and no optimisation])],
    [enable_debug=$enableval], [enable_debug=no])

AC_ARG_ENABLE([sanitize],
    [AS_HELP_STRING([--enable-sanitize], [build with AddressSanitizer + UBSan])],
    [enable_sanitize=$enableval], [enable_sanitize=no])

AC_ARG_ENABLE([tuner],
    [AS_HELP_STRING([--enable-tuner], [also build the tuner binary (bin/tuna)])],
    [enable_tuner=$enableval], [enable_tuner=no])

AC_ARG_ENABLE([eval-debug],
    [AS_HELP_STRING([--enable-eval-debug], [build with evaluation debugging])],
    [enable_eval_debug=$enableval], [enable_eval_debug=no])

# ── Superoptimisation: Souper + Minotaur ─────────────────────────
#
# When --enable-superopt is given, each .c file is compiled to LLVM
# bitcode with clang, then run through two LLVM opt pass plugins:
#
#   • libSouperPass.so  — Souper's optimistic-substitution pass.
#     The pass extracts optimisation candidates from the bitcode and
#     asks an SMT solver (via the `souper` binary, which it shells out
#     to at runtime) whether each candidate has a cheaper equivalent.
#   • libMinotaurPass.so — Minotaur's pass pipeline.
#
# Both passes are loaded by the standard LLVM `opt` driver via
# `-load-pass-plugin <plugin.so>`.  The `souper` binary itself is NOT
# invoked directly by the Makefile — it is invoked by the Souper pass
# at runtime, so we expose its path to the pass via the SOUPER_BIN
# environment variable.
#
# Per-stage fallback: if either pass fails on a particular file, that
# file is still built using the previous stage's bitcode.  See
# build-aux/superopt-stage.sh for the wrapper.
#
# Required tools (all checked at configure time when --enable-superopt):
#   • clang        — emit LLVM bitcode from C source
#   • opt          — the LLVM optimiser driver that loads the pass plugins
#   • llc          — codegen from optimised bitcode to assembly
#   • libSouperPass.so   — Souper's pass plugin
#   • libMinotaurPass.so — Minotaur's pass plugin
#   • souper       — the Souper binary (invoked at runtime by the pass;
#                    exposed via SOUPER_BIN env var so the pass finds it)
AC_ARG_ENABLE([superopt],
    [AS_HELP_STRING([--enable-superopt],
        [enable Souper + Minotaur superoptimisation passes (requires clang, opt, llc, libSouperPass.so, libMinotaurPass.so, souper)])],
    [enable_superopt=$enableval], [enable_superopt=no])

AC_ARG_WITH([clang],
    [AS_HELP_STRING([--with-clang=PATH], [path to clang binary (for bitcode emission)])],
    [CLANG_BIN=$withval], [CLANG_BIN=])
AC_ARG_WITH([opt],
    [AS_HELP_STRING([--with-opt=PATH], [path to LLVM opt binary (the pass driver)])],
    [OPT_BIN=$withval], [OPT_BIN=])
AC_ARG_WITH([llc],
    [AS_HELP_STRING([--with-llc=PATH], [path to llc binary (for codegen)])],
    [LLC_BIN=$withval], [LLC_BIN=])
AC_ARG_WITH([souper-pass],
    [AS_HELP_STRING([--with-souper-pass=PATH],
        [path to libSouperPass.so (Souper's LLVM opt pass plugin)])],
    [SOUPER_PASS=$withval], [SOUPER_PASS=])
AC_ARG_WITH([minotaur-pass],
    [AS_HELP_STRING([--with-minotaur-pass=PATH],
        [path to libMinotaurPass.so (Minotaur's LLVM opt pass plugin)])],
    [MINOTAUR_PASS=$withval], [MINOTAUR_PASS=])
AC_ARG_WITH([souper],
    [AS_HELP_STRING([--with-souper=PATH],
        [path to the souper binary (invoked at runtime by the Souper pass;
         if unset, the pass searches PATH)])],
    [SOUPER_BIN=$withval], [SOUPER_BIN=])

if test "x$enable_superopt" = "xyes"; then
    AC_MSG_NOTICE([Superoptimisation enabled — checking for required tools])

    # clang — required for emitting LLVM bitcode from C source.
    if test -z "$CLANG_BIN"; then
        AC_PATH_PROG([CLANG_BIN_AUTO], [clang], [no])
        CLANG_BIN=$CLANG_BIN_AUTO
    fi
    if test "x$CLANG_BIN" = "xno" -o -z "$CLANG_BIN"; then
        AC_MSG_ERROR([--enable-superopt requires clang (use --with-clang=PATH to specify)])
    fi
    AC_MSG_NOTICE([  clang:           $CLANG_BIN])

    # opt — the LLVM optimiser driver.  Loads pass plugins via
    # -load-pass-plugin.  Must be from the same LLVM release as the
    # plugins were built against, or it will refuse to load them.
    if test -z "$OPT_BIN"; then
        AC_PATH_PROG([OPT_BIN_AUTO], [opt], [no])
        OPT_BIN=$OPT_BIN_AUTO
    fi
    if test "x$OPT_BIN" = "xno" -o -z "$OPT_BIN"; then
        AC_MSG_ERROR([--enable-superopt requires opt (use --with-opt=PATH to specify)])
    fi
    AC_MSG_NOTICE([  opt:             $OPT_BIN])

    # llc — required for codegen from optimised bitcode to assembly.
    if test -z "$LLC_BIN"; then
        AC_PATH_PROG([LLC_BIN_AUTO], [llc], [no])
        LLC_BIN=$LLC_BIN_AUTO
    fi
    if test "x$LLC_BIN" = "xno" -o -z "$LLC_BIN"; then
        AC_MSG_ERROR([--enable-superopt requires llc (use --with-llc=PATH to specify)])
    fi
    AC_MSG_NOTICE([  llc:             $LLC_BIN])

    # libSouperPass.so — Souper's opt pass plugin.  Not a binary, so
    # we check for file existence rather than AC_PATH_PROG.
    if test -z "$SOUPER_PASS"; then
        # Try a few common locations relative to the opt binary's prefix.
        opt_prefix=`AS_DIRNAME(["$OPT_BIN"]) 2>/dev/null`
        for candidate in \
            "$opt_prefix/../lib/libSouperPass.so" \
            "$opt_prefix/../lib64/libSouperPass.so" \
            "/usr/lib/libSouperPass.so" \
            "/usr/local/lib/libSouperPass.so"
        do
            if test -f "$candidate"; then
                SOUPER_PASS="$candidate"
                break
            fi
        done
    fi
    if test -z "$SOUPER_PASS" -o ! -f "$SOUPER_PASS"; then
        AC_MSG_ERROR([--enable-superopt requires libSouperPass.so (use --with-souper-pass=PATH to specify)])
    fi
    AC_MSG_NOTICE([  souper pass:     $SOUPER_PASS])

    # libMinotaurPass.so — Minotaur's opt pass plugin.  Same handling.
    if test -z "$MINOTAUR_PASS"; then
        opt_prefix=`AS_DIRNAME(["$OPT_BIN"]) 2>/dev/null`
        for candidate in \
            "$opt_prefix/../lib/libMinotaurPass.so" \
            "$opt_prefix/../lib64/libMinotaurPass.so" \
            "/usr/lib/libMinotaurPass.so" \
            "/usr/local/lib/libMinotaurPass.so"
        do
            if test -f "$candidate"; then
                MINOTAUR_PASS="$candidate"
                break
            fi
        done
    fi
    if test -z "$MINOTAUR_PASS" -o ! -f "$MINOTAUR_PASS"; then
        AC_MSG_ERROR([--enable-superopt requires libMinotaurPass.so (use --with-minotaur-pass=PATH to specify)])
    fi
    AC_MSG_NOTICE([  minotaur pass:   $MINOTAUR_PASS])

    # souper binary — invoked at runtime by the Souper pass.  Optional:
    # if not given, the pass searches PATH at runtime.  We still check
    # for it at configure time so we can warn early if it's missing.
    if test -z "$SOUPER_BIN"; then
        AC_PATH_PROG([SOUPER_BIN_AUTO], [souper], [no])
        SOUPER_BIN=$SOUPER_BIN_AUTO
    fi
    if test "x$SOUPER_BIN" = "xno" -o -z "$SOUPER_BIN"; then
        AC_MSG_WARN([souper binary not found on PATH — the Souper pass will need to find it at runtime via SOUPER_BIN or PATH (use --with-souper=PATH to specify at configure time)])
        SOUPER_BIN=""
    else
        AC_MSG_NOTICE([  souper binary:   $SOUPER_BIN])
    fi

    AC_SUBST([CLANG_BIN])
    AC_SUBST([OPT_BIN])
    AC_SUBST([LLC_BIN])
    AC_SUBST([SOUPER_PASS])
    AC_SUBST([MINOTAUR_PASS])
    AC_SUBST([SOUPER_BIN])
fi
AM_CONDITIONAL([ENABLE_SUPEROPT], [test "x$enable_superopt" = "xyes"])

AM_CONDITIONAL([ENABLE_TUNER], [test "x$enable_tuner" = "xyes"])

# Set CFLAGS / LDFLAGS based on selected mode
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

if test "x$enable_eval_debug" = "xyes"; then
    CFLAGS="$CFLAGS -DEVAL_DEBUG"
    AC_MSG_NOTICE([Evaluation debug enabled])
fi

AM_CONDITIONAL([ENABLE_EVAL_DEBUG], [test "x$enable_eval_debug" = "xyes"])

# Generate files
AC_CONFIG_FILES([Makefile src/Makefile])
AC_OUTPUT

AC_MSG_NOTICE([
-------------------------------------------
  Salmon Chess Engine — configuration done
-------------------------------------------
  Now run:  make
            make tuner   (if --enable-tuner)
            make perft   (smoke test)

  Superoptimisation: $enable_superopt])

if test "x$enable_superopt" = "xyes"; then
    AC_MSG_NOTICE([    clang=$CLANG_BIN
    opt=$OPT_BIN
    llc=$LLC_BIN
    souper-pass=$SOUPER_PASS
    minotaur-pass=$MINOTAUR_PASS
    souper=$SOUPER_BIN])
fi
