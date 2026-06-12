#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RMUA2026 stage2 避障日志解析器（阶段0.2）

从 ROS 日志（rosout 文本 / docker logs 输出 / rosbag echo）解析:
  [STAGE2_PATH] ...   规划+感知+控制每帧指标
  [PERF] ...          控制环耗时/帧率
  [PATH] ...          stage1 跟踪指标(可选)

用法:
  # 直接喂 docker logs
  docker logs my_code 2>&1 | python3 parse_stage2_log.py --out /tmp/run.csv

  # 或解析已保存的日志文件
  python3 parse_stage2_log.py --in /tmp/flight.log --out /tmp/run.csv

输出:
  - 每帧 CSV(--out)
  - stderr 打印汇总指标(反应提前量/可行率/fallback频率/横向偏离/帧率等)
"""
import sys
import re
import csv
import argparse

# 通用 key=value 解析（对字段顺序/增删鲁棒）
KV = re.compile(r'(\w+)=(-?\d+\.?\d*(?:[eE][+-]?\d+)?|inf|-inf|nan)')

def parse_kv(line):
    """提取一行里所有 key=value（数值型）。cmd=(...)/lim=(...) 这类括号值单独处理。"""
    d = {}
    for k, v in KV.findall(line):
        if v in ('inf', '-inf', 'nan'):
            d[k] = float('inf') if v != 'nan' else float('nan')
            if v == '-inf':
                d[k] = float('-inf')
        else:
            try:
                d[k] = float(v)
            except ValueError:
                pass
    # cmd=(vx vy vz yaw)
    m = re.search(r'cmd=\(([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+)\)', line)
    if m:
        d['cmd_vx'], d['cmd_vy'], d['cmd_vz'], d['cmd_yaw'] = map(float, m.groups())
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='infile', default=None, help='输入日志文件，缺省读 stdin')
    ap.add_argument('--out', dest='outfile', default='/tmp/stage2_metrics.csv')
    args = ap.parse_args()

    src = open(args.infile) if args.infile else sys.stdin

    rows = []          # STAGE2_PATH 每帧
    perf_rows = []     # PERF 每帧
    for raw in src:
        line = raw.rstrip('\n')
        if '[STAGE2_PATH]' in line:
            d = parse_kv(line)
            # mode 是字符串，单独取
            mm = re.search(r'mode=(\w+)', line)
            d['mode'] = mm.group(1) if mm else ''
            rows.append(d)
        elif '[PERF]' in line:
            perf_rows.append(parse_kv(line))

    if not rows:
        sys.stderr.write('警告: 未解析到任何 [STAGE2_PATH] 行。stage2 是否进入? 日志是否完整?\n')

    # 写每帧 CSV
    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        # 把常用列提前
        front = ['mode', 'vh', 'obs', 'trigger', 'cdev', 'feas', 'tot', 'fb',
                 'latched', 'dyn', 'closing', 'ttc', 'cmd_vx', 'cmd_vy', 'cmd_vz']
        ordered = [k for k in front if k in keys] + [k for k in keys if k not in front]
        with open(args.outfile, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=ordered)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        sys.stderr.write(f'已写 {len(rows)} 帧 → {args.outfile}\n')

    # ---- 汇总衍生指标 ----
    def col(rows, k):
        return [r[k] for r in rows if k in r and r[k] == r[k]]  # 排除 nan

    sys.stderr.write('\n========== 汇总指标 ==========\n')

    # 帧率
    fps = col(perf_rows, 'fps_avg')
    cb_ms = col(perf_rows, 'cb_ms')
    if fps:
        sys.stderr.write(f'控制帧率 fps_avg: 末值={fps[-1]:.1f} 最低={min(fps):.1f} (目标≥33)\n')
    if cb_ms:
        sys.stderr.write(f'单帧耗时 cb_ms: 均值={sum(cb_ms)/len(cb_ms):.2f} 最大={max(cb_ms):.2f} ms\n')

    # 横向偏离
    cdev = col(rows, 'cdev')
    if cdev:
        rms = (sum(x*x for x in cdev)/len(cdev))**0.5
        sys.stderr.write(f'横向偏离中心线 cdev: max={max(cdev):.2f} RMS={rms:.2f} m  (越大越接近被吸下去)\n')

    # 反应提前量 = trigger / vh（只在 AVOID 或有障碍时有意义）
    lead = []
    for r in rows:
        vh = r.get('vh', 0.0)
        trig = r.get('trigger', 0.0)
        if vh and vh > 0.5:
            lead.append(trig / vh)
    if lead:
        sys.stderr.write(f'反应提前量 trigger/vh: min={min(lead):.2f}s 均值={sum(lead)/len(lead):.2f}s (越大越及时)\n')

    # 候选可行率 + fallback 频率（只看真正调用了规划的帧：tot>0）
    plan_frames = [r for r in rows if r.get('tot', 0) > 0]
    if plan_frames:
        feas_ratio = [r['feas']/r['tot'] for r in plan_frames if r.get('tot', 0) > 0]
        fb_cnt = sum(1 for r in plan_frames if r.get('fb', 0) >= 1)
        sys.stderr.write(f'候选可行率 feas/tot: 均值={sum(feas_ratio)/len(feas_ratio):.2f} 最低={min(feas_ratio):.2f}\n')
        sys.stderr.write(f'fallback 触发: {fb_cnt}/{len(plan_frames)} 规划帧 ({100.0*fb_cnt/len(plan_frames):.1f}%) (越低越好,fallback=盲飞)\n')

    # AVOID 占比 + latch 占比
    n = len(rows)
    if n:
        avoid = sum(1 for r in rows if r.get('mode') == 'AVOID')
        latch = sum(1 for r in rows if r.get('latched', 0) >= 1)
        dyn = sum(1 for r in rows if r.get('dyn', 0) >= 1)
        sys.stderr.write(f'AVOID占比={100.0*avoid/n:.1f}%  latch占比={100.0*latch/n:.1f}%  动态障碍占比={100.0*dyn/n:.1f}%\n')

    # 最小障碍间距
    obs = [r['obs'] for r in rows if 'obs' in r and r['obs'] == r['obs'] and r['obs'] < 1e6]
    if obs:
        sys.stderr.write(f'最小障碍间距 obs: min={min(obs):.2f} m\n')

    sys.stderr.write('==============================\n')

if __name__ == '__main__':
    main()
