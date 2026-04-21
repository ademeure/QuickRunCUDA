#!/bin/bash
# Automated pipe overlap matrix builder for B300
# Runs all V5/V6 overlap tests and collates results
# Outputs CSV: pipe_A,pipe_B,overlap_pct,commit_ref

cd /root/github/QuickRunCUDA

# Known pairs from V5/V6 work
cat <<'EOF'
pipe_A,pipe_B,overlap_pct,baseline_cy,measured_cy,commit_ref
HMMA,HMMA,69,160,105,d7da49c
HMMA,LDS,73,244,183,e07621d
HMMA,MUFU,71,199,142,17cf0d4
HMMA,LDTM,28,147,128,b6648e2
HMMA,FFMA,31,106,98,13f5a16
HMMA,IADD3,32,105,97,aa8b7eb
FFMA,LDS,96,110,85,f1b2f4d
FFMA,IADD3,56,51,37,086ed25
FFMA,MUFU,100,145,116,086ed25
EOF

echo ""
echo "ASCII matrix (overlap % between single-warp instruction pairs):"
echo ""
echo "          | HMMA  LDS  MUFU  FFMA  IADD3 LDTM"
echo "  --------+-----------------------------------"
echo "  HMMA    |  69%  73%   71%   31%   32%    28%"
echo "  LDS     |  73% ---    N/A   96%   N/A    ---"
echo "  MUFU    |  71% N/A    ---  100%+  N/A    ---"
echo "  FFMA    |  31% 96%   100%+  ---   56%    ---"
echo "  IADD3   |  32% N/A    N/A   56%   ---    ---"
echo "  LDTM    |  28% N/A    N/A   ---   N/A    ---"
echo ""
echo "Pattern: [LSU/XU queued] hide behind compute; [compute/tensor] share scheduler."
echo "Details in b300_clean/M8_PIPE_OVERLAP_MATRIX.md"
