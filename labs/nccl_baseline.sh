#!/usr/bin/env bash
# Ch18 §18.4 nccl 一键基线:构建 nccl-tests -> all_reduce + alltoallv(节点内/跨机) -> busbw 曲线 + 验收判读。
#
# 验收标准(Ch18 §18.4 表):期望 busbw = 0.875 x 链路带宽(N=8;ring all-reduce 系数 2(N-1)/N)
#   节点内 NVLink 900GB/s  -> 期望 ~780 GB/s
#   跨机 IB NDR 8x50GB/s rail-optimized -> 大消息聚合后按 NIC 数近似线性 -> 期望 ~350 GB/s
#   低于期望 20% 以上 => 拓扑/配置问题(不是"正常波动"),四类故障排查见 Ch18 §18.4:
#   ① RoCE PFC 风暴 ② GDR 失效退回 CPU 拷贝 ③ NUMA/跨 socket ④ MTU 不一致
#
# 用法:
#   ./nccl_baseline.sh                      # 节点内 8 卡
#   ./nccl_baseline.sh --hostfile nodes.txt # 跨机(hostfile 每行一个节点,免密 ssh)
#   ./nccl_baseline.sh --link-bw 400        # 覆盖链路带宽(GB/s),期望 = 0.875 x 400
#   ./nccl_baseline.sh --expected 350       # 直接指定期望 busbw(优先级最高)
#   ./nccl_baseline.sh --debug              # NCCL_DEBUG=INFO(排障用,输出很大)
# 依赖:git、make、OpenMPI 或 MPICH(mpirun/hydra)、CUDA 环境;nccl-tests 自动 clone 到 --build-dir。
set -euo pipefail

GPUS=8
HOSTFILE=""
BUILD_DIR="$(pwd)/nccl-build"
LINK_BW_GB=""
EXPECTED_GB=""
DEBUG=0

usage() {
  sed -n '2,16p' "$0"
  echo "用法: $0 [--gpus N] [--hostfile f] [--build-dir d] [--link-bw GB] [--expected GB] [--debug]"
  exit 0
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2;;
    --hostfile) HOSTFILE="$2"; shift 2;;
    --build-dir) BUILD_DIR="$2"; shift 2;;
    --link-bw) LINK_BW_GB="$2"; shift 2;;
    --expected) EXPECTED_GB="$2"; shift 2;;
    --debug) DEBUG=1; shift;;
    -h|--help) usage;;
    *) echo "未知参数: $1"; usage;;
  esac
done

[[ "$GPUS" -eq 8 ]] || echo "注意: 期望系数 0.875 按 N=8 推导(Ch18 §18.4);GPUS=$GPUS 时判读需按 2(N-1)/N 自行换算"

# ---------- 1. MPI 启动器 ----------
if command -v mpirun >/dev/null 2>&1; then MPIRUN=mpirun
elif command -v hydra  >/dev/null 2>&1; then MPIRUN=hydra
else echo "未找到 mpirun/hydra:安装 OpenMPI 或 MPICH(跨机需免密 ssh)"; exit 1; fi

# ---------- 2. 构建 nccl-tests(已构建则跳过) ----------
SRC="$BUILD_DIR/nccl-tests"
if [[ ! -f "$SRC/src/all_reduce_perf" ]]; then
  echo "[1/3] 构建 nccl-tests -> $SRC"
  mkdir -p "$BUILD_DIR"
  [[ -d "$SRC" ]] || git clone --depth 1 https://github.com/NVIDIA/nccl-tests "$SRC"
  make -C "$SRC" -j"$(nproc)" MPI=1
else
  echo "[1/3] nccl-tests 已存在,跳过构建"
fi
BIN="$SRC/src"

# ---------- 3. 运行 + 解析 ----------
# nccl-tests 数据行: size(B) count time time algbw busbw #wrong time algbw busbw #wrong
# out-of-place busbw = 第 6 列(in-place 为第 10 列,这里不取)
# 消息范围与 Ch18 §18.4 一致: -b 8M -e 8G -f 2(2 的幂,8 级)
parse() { # $1 = 原始日志 -> busbw 曲线 + 末行 "MAX <值>"
  awk '
    function human(b) {
      if (b >= 1073741824) return b/1073741824 "G"
      if (b >= 1048576)   return b/1048576 "M"
      if (b >= 1024)      return b/1024 "K"
      return b ""
    }
    $1 ~ /^[0-9]+$/ && NF >= 10 {
      printf "  %-8s %12.2f\n", human($1), $6
      if ($6 > max) max = $6
    }
    END { if (max > 0) printf "MAX %.2f\n", max; else print "MAX 0" }
  ' "$1"
}

run_one() { # $1 = 测试程序名, $2 = 日志文件
  local name="$1" log="$2"
  if [[ -n "$HOSTFILE" ]]; then
    local nnodes total
    nnodes=$(grep -v '^[[:space:]]*#' "$HOSTFILE" | grep -c . || true)
    total=$(( nnodes * GPUS ))
    if [[ "$MPIRUN" == mpirun ]]; then
      mpirun -f "$HOSTFILE" -npernode "$GPUS" -np "$total" \
        "$BIN/$name" -b 8M -e 8G -f 2 -g 1 > "$log" 2>&1
    else
      local hosts
      hosts=$(grep -v '^[[:space:]]*#' "$HOSTFILE" | paste -sd, -)
      hydra -np "$total" -host "$hosts" \
        "$BIN/$name" -b 8M -e 8G -f 2 -g 1 > "$log" 2>&1
    fi
  else
    local extra=()
    if [[ "$MPIRUN" == mpirun ]] && [[ "$(id -u)" == "0" ]] \
       && mpirun --version 2>/dev/null | grep -qi openmpi; then
      extra=(--allow-run-as-root)
    fi
    "${MPIRUN[@]}" ${extra[@]+"${extra[@]}"} -np "$GPUS" \
      "$BIN/$name" -b 8M -e 8G -f 2 -g 1 > "$log" 2>&1
  fi
}

if [[ "$DEBUG" -eq 1 ]]; then export NCCL_DEBUG=INFO; echo "(NCCL_DEBUG=INFO)"; fi

echo "[2/3] 运行测试 (MPI=$MPIRUN, GPUS/节点=$GPUS, 范围 8M-8G)"
LOGDIR="$BUILD_DIR/logs-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOGDIR"
run_one all_reduce_perf "$LOGDIR/all_reduce.log"
run_one alltoallv_perf  "$LOGDIR/alltoallv.log"

# 期望 busbw: --expected > --link-bw x 0.875 > 默认(节点内 780 / 跨机 rail 350,Ch18 §18.4 表)
if [[ -z "$EXPECTED_GB" ]]; then
  if [[ -n "$LINK_BW_GB" ]]; then
    EXPECTED_GB=$(awk -v l="$LINK_BW_GB" 'BEGIN{printf "%.1f", 0.875*l}')
  elif [[ -n "$HOSTFILE" ]]; then
    EXPECTED_GB=350
  else
    EXPECTED_GB=780
  fi
fi

echo "[3/3] 结果(out-of-place busbw,GB/s;期望 ~$EXPECTED_GB)"
overall=0
for t in all_reduce alltoallv; do
  echo "== $t =="
  out="$(parse "$LOGDIR/${t}.log")"
  echo "$out" | sed '$d'
  mx="$(echo "$out" | tail -1 | awk '{print $2}')"
  verdict="$(awk -v a="$mx" -v e="$EXPECTED_GB" 'BEGIN{print (a+0 >= 0.8*e) ? "PASS" : "FAIL"}')"
  [[ "$verdict" == PASS ]] || overall=1
  echo "  -> 最大 busbw = ${mx} GB/s, 期望 ~${EXPECTED_GB} GB/s (0.875 x 链路, N=8): $verdict"
done

if [[ "$overall" -eq 0 ]]; then
  echo "结论: 基线达标。注意:nccl-tests 是空载集合通信——合格是生产低时延的必要条件,不是充分条件(Ch18 §18.4 踩坑框)"
else
  echo "结论: 低于期望 20%+ => 拓扑/配置问题。排查顺序(Ch18 §18.4 四类故障):"
  echo "  1) nvidia-smi topo -m 看 NVLink 是否降档(H800 坑,附录 B)"
  echo "  2) NCCL_DEBUG=INFO 看 GDR/transport 状态(--debug)"
  echo "  3) numactl --hardware 查 NIC/GPU 亲和;ibstat 查 active_MTU"
  echo "  4) RoCE: ethtool -S 查 PFC/pause 计数"
  echo "原始日志: $LOGDIR (all_reduce.log / alltoallv.log)"
fi
