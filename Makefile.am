SUBDIRS = src

# Files that should be in the distribution tarball but aren't picked up
# automatically by automake's _SOURCES / _HEADERS detection.
EXTRA_DIST = \
    configure.ac \
    Makefile.am \
    src/Makefile.am \
    README.md \
    LICENSE.md \
    CONTRIBUTING.md \
    TUNAD.md \
    supervised_tune.py \
    tests/test_mate.py \
    tests/mate_positions.csv \
    src/nnue.bin

# Keep the perft smoke-test target available from the top level.
# Position 5 from CPW, depth 3, expected 62379.
.PHONY: perft
perft: all
	@echo "Running perft depth 3 on position 5..."
	@echo -e "position fen rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8\nperft 3\nquit" \
            | ./src/engine

# Remove generated autotools files with 'make maintainer-clean'
MAINTAINERCLEANFILES = \
    Makefile.in aclocal.m4 compile config.guess config.h.in \
    config.sub configure depcomp install-sh missing