#!/bin/bash
# superopt-stage.sh — run one superoptimisation stage on an LLVM bitcode file.
#
# Usage:
#   superopt-stage.sh <stage_name> <tool_command...> <input.bc> <output.bc>
#
# Where:
#   stage_name       — human-readable label for log messages ("souper", "minotaur")
#   tool_command...  — the tool and its flags, ending with the input file
#                      (e.g. "souper -z3-behind=true foo.bc")
#   input.bc         — the input bitcode file
#   output.bc        — where to write the optimised bitcode
#
# Behaviour:
#   • Runs `<tool_command> -o <output.bc>` (we append -o and the output path).
#   • On success: prints "  [stage_name] OK: input.bc -> output.bc" and exits 0.
#   • On failure: prints "  [stage_name] FAIL: <reason>; falling back to input",
#                  copies input.bc to output.bc, and exits 0 (so the Makefile
#                  rule's downstream stages still run on the best-available IR).
#   • The fallback is per-stage: a Souper failure does NOT prevent Minotaur
#     from running on the un-Souper'd bitcode, and vice versa.
#
# This script is intentionally tool-agnostic — the Makefile wraps each
# superopt tool (Souper, Minotaur) in a separate call so the fallback
# logic is uniform.

set -u  # error on undefined variables; do NOT use -e (we catch failures ourselves)

stage_name="$1"; shift

# Expected layout of remaining args:
#   <tool> [tool-flags...] <input.bc> <output.bc>
# The tool command is everything EXCEPT the last two args (output_bc and input_bc).
# We then invoke "<tool> [tool-flags...] <input.bc> -o <output.bc>".
# (Both Souper and LLVM-style opt tools accept `-o <path>` as the output flag.)
if [ "$#" -lt 3 ]; then
    echo "  [$stage_name] FAIL: not enough arguments (need at least stage + tool + input + output)" >&2
    exit 2
fi
output_bc="${!#}"                # last arg = output
input_bc="${@:(-2):1}"           # second-to-last = input
# tool_cmd = everything from index 1 up to and including the input file
# (i.e. everything except the last arg, which is the output).
tool_cmd=("${@:1:$#-1}")

if [ ! -f "$input_bc" ]; then
    echo "  [$stage_name] FAIL: input '$input_bc' does not exist" >&2
    exit 2
fi

# Run the tool.  We append "-o <output_bc>" to the tool command — both
# Souper and LLVM opt-style tools accept -o as the output flag.  If a
# tool needs a different flag, the caller can wrap it.
#
# We use a temporary file for the actual output and only move it into
# place on success, so a partial/corrupt output never replaces a good
# fallback.  This matters because some tools write a partial file
# before crashing.
tmp_out="${output_bc}.tmp.$$"
rm -f "$tmp_out"

# shellcheck disable=SC2086  # we want tool_cmd to word-split
"${tool_cmd[@]}" -o "$tmp_out" 2> "${output_bc}.log"
rc=$?

if [ "$rc" -ne 0 ] || [ ! -s "$tmp_out" ]; then
    reason="exit code $rc"
    [ ! -s "$tmp_out" ] && reason="empty output"
    echo "  [$stage_name] FAIL: $reason on $(basename "$input_bc"); falling back to input (see ${output_bc}.log)" >&2
    # Fallback: copy the input to the output so downstream stages have something to chew on.
    cp "$input_bc" "$output_bc"
    rm -f "$tmp_out"
    exit 0
fi

mv "$tmp_out" "$output_bc"
rm -f "${output_bc}.log"
echo "  [$stage_name] OK: $(basename "$input_bc") -> $(basename "$output_bc")"
exit 0
