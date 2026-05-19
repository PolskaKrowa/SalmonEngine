# -------------------------------------------------------------
#  configure.ac — Salmon Chess Engine
#
#  Run:  autoreconf --install
#  Then: ./configure [--prefix=/usr/local] [--enable-debug]
#                    [--enable-sanitize] [--enable-tuner]
# -------------------------------------------------------------

AC_INIT([salmon], [2.0], [], [salmon-engine])
AC_CONFIG_SRCDIR([src/main.c])          # sanity-check: fails if this file is missing
AC_CONFIG_HEADERS([include/config.h])   # generates a config.h with detected defines
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_MACRO_DIR([m4])
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
])
