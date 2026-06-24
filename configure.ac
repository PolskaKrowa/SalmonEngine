# -------------------------------------------------------------
#  configure.ac — Salmon Chess Engine
#
#  Run:  autoreconf --install
#  Then: ./configure [--prefix=/usr/local] [--enable-debug]
#                    [--enable-sanitize] [--enable-tuner]
#                    [--enable-eval-debug]
#                    [--enable-superopt]
#                    [--with-souper=PATH] [--with-minotaur=PATH]
#                    [--with-clang=PATH] [--with-llc=PATH]
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
# bitcode with clang, then run through Souper (optimistic substitution
# with SMT solving) and Minotaur (a custom LLVM opt pass pipeline).
# Both stages have per-stage fallbacks: if either tool fails on a
# particular file, that file is still built using the previous stage's
# bitcode.  See build-aux/superopt-stage.sh for the wrapper.
#
# Requires: clang (for bitcode emission + final codegen), llc (for
# codegen if clang is not used for that step), souper, minotaur.
AC_ARG_ENABLE([superopt],
    [AS_HELP_STRING([--enable-superopt],
        [enable Souper + Minotaur superoptimisation passes (requires clang, llc, souper, minotaur)])],
    [enable_superopt=$enableval], [enable_superopt=no])

AC_ARG_WITH([souper],
    [AS_HELP_STRING([--with-souper=PATH], [path to souper binary])],
    [SOUPER_BIN=$withval], [SOUPER_BIN=])
AC_ARG_WITH([minotaur],
    [AS_HELP_STRING([--with-minotaur=PATH], [path to minotaur binary])],
    [MINOTAUR_BIN=$withval], [MINOTAUR_BIN=])
AC_ARG_WITH([clang],
    [AS_HELP_STRING([--with-clang=PATH], [path to clang binary (for bitcode emission)])],
    [CLANG_BIN=$withval], [CLANG_BIN=])
AC_ARG_WITH([llc],
    [AS_HELP_STRING([--with-llc=PATH], [path to llc binary])],
    [LLC_BIN=$withval], [LLC_BIN=])

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
    AC_MSG_NOTICE([  clang:    $CLANG_BIN])

    # llc — required for codegen from optimised bitcode to assembly.
    if test -z "$LLC_BIN"; then
        AC_PATH_PROG([LLC_BIN_AUTO], [llc], [no])
        LLC_BIN=$LLC_BIN_AUTO
    fi
    if test "x$LLC_BIN" = "xno" -o -z "$LLC_BIN"; then
        AC_MSG_ERROR([--enable-superopt requires llc (use --with-llc=PATH to specify)])
    fi
    AC_MSG_NOTICE([  llc:      $LLC_BIN])

    # souper — required (no fallback at the build-system level; per-file
    # fallback is handled by superopt-stage.sh).
    if test -z "$SOUPER_BIN"; then
        AC_PATH_PROG([SOUPER_BIN_AUTO], [souper], [no])
        SOUPER_BIN=$SOUPER_BIN_AUTO
    fi
    if test "x$SOUPER_BIN" = "xno" -o -z "$SOUPER_BIN"; then
        AC_MSG_ERROR([--enable-superopt requires souper (use --with-souper=PATH to specify)])
    fi
    AC_MSG_NOTICE([  souper:   $SOUPER_BIN])

    # minotaur — same as souper.
    if test -z "$MINOTAUR_BIN"; then
        AC_PATH_PROG([MINOTAUR_BIN_AUTO], [minotaur], [no])
        MINOTAUR_BIN=$MINOTAUR_BIN_AUTO
    fi
    if test "x$MINOTAUR_BIN" = "xno" -o -z "$MINOTAUR_BIN"; then
        AC_MSG_ERROR([--enable-superopt requires minotaur (use --with-minotaur=PATH to specify)])
    fi
    AC_MSG_NOTICE([  minotaur: $MINOTAUR_BIN])

    AC_SUBST([CLANG_BIN])
    AC_SUBST([LLC_BIN])
    AC_SUBST([SOUPER_BIN])
    AC_SUBST([MINOTAUR_BIN])

    # Override CC/CFLAGS so the automake-generated .c.o rule (which
    # would use gcc) is replaced by our custom superopt rule below.
    # We still keep CC=gcc for the final link step (handled by the
    # automake linker rule, which uses CCLD).
    SUPEROPT_CFLAGS_BASE="$CFLAGS"
    AC_SUBST([SUPEROPT_CFLAGS_BASE])
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
    llc=$LLC_BIN
    souper=$SOUPER_BIN
    minotaur=$MINOTAUR_BIN])
fi
