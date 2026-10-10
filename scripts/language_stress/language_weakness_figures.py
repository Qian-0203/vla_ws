#!/usr/bin/env python3
"""Per-task figures for docs/pi05_language_weakness.md and docs/openvla_language_weakness.md.

Reads every *.jsonl in a results directory (local eval logs, or a download of the HF mirror
Qian0203/vla_ws-eval-results), dedupes rollouts on (task_id, episode_idx), and writes
the per-policy heatmaps and mirror-pair (tasks 7/9) charts to the output directory.

  uv run --with matplotlib --with numpy python scripts/language_stress/language_weakness_figures.py \
      <results_dir> docs/figures
"""
import collections
import glob
import json
import math
import os
import re
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

plt.rcParams["font.family"] = ["DejaVu Sans", "Noto Sans CJK TC"]  # CJK fallback for code-switch labels

RESULTS, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

# (suite, condition, run note) -> {(task_id, episode_idx): row}; shards are merged.
runs = collections.defaultdict(dict)
for f in glob.glob(os.path.join(RESULTS, "*.jsonl")):
    key = re.sub(r"--shard\dof\d$", "", os.path.basename(f)[: -len(".jsonl")]).split("--")
    suite, cond, note = key[0], key[1], "--".join(key[2:])
    for line in open(f):
        if line.strip():
            r = json.loads(line)
            runs[(suite, cond, note)][(r["task_id"], r["episode_idx"])] = r


def per(c, m, s="libero_spatial"):
    """{task_id: (successes, rollouts)} for one run."""
    out = collections.defaultdict(lambda: [0, 0])
    for r in runs.get((s, c, m), {}).values():
        out[r["task_id"]][0] += int(r["success"])
        out[r["task_id"]][1] += 1
    return {t: tuple(v) for t, v in out.items()}


P='pi05_libero'; SC='pi05_libero--screen'
OV='openvla-official-libero-spatial'; OVS=OV+'--screen'


def heatmap(P, SC, title, fname):
    """Condition x task success grid for one policy (P: full-run note, SC: screening note)."""
    rows=[('Baseline','default',P,'default (n=50/task)'),
    ('Content-free','length_control_infix',P,'length_control_infix'),('Content-free','length_control_suffix',P,'length_control_suffix'),
    ('Paraphrase','paraphrase_lexical',P,'paraphrase_lexical'),('Paraphrase','paraphrase_syntactic',P,'paraphrase_syntactic'),
    ('Lexical','noun_synonym',SC,'noun_synonym *'),('Lexical','functional_landmark',SC,'functional_landmark *'),('Lexical','code_switch',SC,'code_switch *'),('Lexical','translate_zh',SC,'translate_zh *'),
    ('Cue type','target_cue_proximity_near',P,'target_cue_proximity_near'),('Cue type','target_cue_landmark',P,'target_cue_landmark'),('Cue type','target_cue_region',P,'target_cue_region'),('Cue type','target_cue_region_v2',P,'target_cue_region_v2'),
    ('Distractor mention','positive_contrast',P,'positive_contrast'),('Distractor mention','negative_contrast',P,'negative_contrast'),('Distractor mention','self_correction',SC,'self_correction *'),
    ('Exclusion','negation_only',P,'negation_only'),('Exclusion','negation_except',SC,'negation_except *'),('Exclusion','negation_other',SC,'negation_other *'),('Exclusion','negation_swap',SC,'negation_swap * (other bowl)'),
    ('Reliance probes','no_location',SC,'no_location *'),('Reliance probes','target_swap',SC,'target_swap * (other bowl)')]
    cols=[0,2,4,8,1,3,5,6,7,9]
    M=np.full((len(rows),len(cols)),np.nan); N=np.zeros_like(M)
    labels=[r[3] for r in rows]
    for i,(_,c,m,_) in enumerate(rows):
        p=per(c,m)
        if not p and m==P and per(c,SC):  # no full run for this policy: fall back to its screen
            p=per(c,SC); labels[i]+=' *'
        for j,t in enumerate(cols):
            if t in p: M[i,j]=p[t][0]/p[t][1]; N[i,j]=p[t][1]
    ramp=['#f7f6f3','#cde2fb','#9ec5f4','#6da7ec','#3987e5','#256abf','#184f95','#0d366b']
    cmap=LinearSegmentedColormap.from_list('b',ramp); cmap.set_bad('#ffffff')
    fig,ax=plt.subplots(figsize=(10.5,9.6),dpi=150)
    ax.imshow(np.ma.masked_invalid(M),cmap=cmap,vmin=0,vmax=1,aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v=M[i,j]
            if np.isnan(v): ax.text(j,i,'—',ha='center',va='center',color='#9a9890',fontsize=8); continue
            ax.text(j,i,f'{v*100:.0f}',ha='center',va='center',fontsize=8.5,color='#ffffff' if v>0.55 else '#2b2a27')
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([f'task {t}' for t in cols],fontsize=8.5,color='#4a4945')
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels,fontsize=8.5,color='#2b2a27')
    ax.xaxis.tick_top()
    for x in (3.5,7.5): ax.axvline(x,color='#2b2a27',lw=1.4)
    prev=None
    for i,r in enumerate(rows):
        if prev and r[0]!=prev: ax.axhline(i-0.5,color='#ffffff',lw=3)
        prev=r[0]
    ax.set_xticks(np.arange(-.5,len(cols)),minor=True); ax.set_yticks(np.arange(-.5,len(rows)),minor=True)
    ax.grid(which='minor',color='#ffffff',lw=1); ax.tick_params(which='both',length=0)
    for s in ax.spines.values(): s.set_visible(False)
    for x,lab in ((1.5,'π0.5 layout-driven'),(5.5,'π0.5 language-read'),(8.5,'mirror pair')):
        ax.text(x,-1.6,lab,ha='center',va='bottom',fontsize=9.5,fontweight='bold',color='#2b2a27')
    # family labels
    prev=None
    for i,r in enumerate(rows):
        if r[0]!=prev:
            k=sum(1 for q in rows if q[0]==r[0])
            ax.text(10.0,i+(k-1)/2,r[0],ha='left',va='center',fontsize=8.5,color='#6b6a64')
        prev=r[0]
    ax.set_title(title,fontsize=12,loc='left',pad=62,color='#2b2a27')
    fig.text(0.01,0.01,'* screening run, 5 trials/task (±20 pts per cell); unmarked rows 50 trials/task. "(other bowl)": success scored on the bowl the prompt names, not the trained target.\nTasks 7 and 9 share one layout (bowls on the stove and on the cabinet top) with the target swapped, so only language can separate them.',fontsize=7.5,color='#6b6a64')
    plt.subplots_adjust(left=0.27,right=0.86,top=0.86,bottom=0.07)
    plt.savefig(os.path.join(OUT, fname)); plt.close()


heatmap(P, SC, 'π0.5-LIBERO success rate (%) by prompt condition and task', 'pi05_language_heatmap.png')
heatmap(OV, OVS, 'OpenVLA (official checkpoint) success rate (%) by condition and task',
        'openvla_official_language_heatmap.png')

# mirror-pair gradient
items=[('default',P,'exact training wording'),('target_cue_proximity_near',P,'"near the stove" (relation synonym)'),('noun_synonym',SC,'"on the cooktop" (noun synonym)'),('target_cue_landmark',P,'"next to the stove"'),('negative_contrast',P,'target + "not the one on …"'),('functional_landmark',SC,'"the appliance you cook on"'),('no_location',SC,'location removed (chance ≈ 50%)'),('code_switch',SC,'"on the 炉子" (code-switch)'),('translate_zh',SC,'full Chinese'),('self_correction',SC,'distractor first, then "sorry, I mean …"'),('target_cue_region',P,'table-region cue ("front-left")'),('negation_only',P,'"the bowl that is not on …"')]
def wil(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
fig,ax=plt.subplots(figsize=(9,5.6),dpi=150)
ys=np.arange(len(items))[::-1]
for y,(c,m,lab) in zip(ys,items):
    p=per(c,m); k=p[7][0]+p[9][0]; n=p[7][1]+p[9][1]; lo,hi=wil(k,n); v=k/n
    ax.plot([lo*100,hi*100],[y,y],color='#9ec5f4',lw=2,solid_capstyle='round',zorder=1)
    ax.scatter([v*100],[y],s=46,color='#256abf',zorder=2,edgecolors='#ffffff',linewidths=1.5)
    ax.text(102,y,f'{v*100:.0f}%  ({k}/{n})',va='center',fontsize=8.5,color='#2b2a27')
ax.axvline(50,color='#9a9890',lw=1,ls=(0,(3,3)))
ax.set_yticks(ys); ax.set_yticklabels([i[2] for i in items],fontsize=8.5,color='#2b2a27')
ax.set_xlim(0,100); ax.set_xlabel('success rate on tasks 7 + 9 (%), 95% Wilson CI',fontsize=8.5,color='#4a4945')
for s in ('top','right','left'): ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color('#c9c7c0'); ax.tick_params(axis='y',length=0); ax.tick_params(axis='x',colors='#6b6a64',labelsize=8)
ax.grid(axis='x',color='#ecebe7',lw=0.8); ax.set_axisbelow(True)
fig.text(0.02,0.95,'Mirror pair (tasks 7/9): the only place layout cannot pick the bowl',fontsize=11.5,color='#2b2a27')
plt.subplots_adjust(left=0.38,right=0.86,top=0.88,bottom=0.1)
plt.savefig(OUT+'/pi05_mirror_pair.png'); plt.close()

# Mirror pair, pi05 vs. official OpenVLA, conditions both policies ran.
items = [('default', '', 'exact training wording'), ('paraphrase_syntactic', '', 'syntactic paraphrase'),
         ('paraphrase_lexical', '', 'lexical paraphrase'), ('length_control_infix', '', 'content-free infix'),
         ('length_control_suffix', '', 'content-free suffix'), ('target_cue_proximity_near', '', '"near the stove"'),
         ('target_cue_landmark', '', '"next to the stove"'), ('noun_synonym', 's', '"on the cooktop" *'),
         ('functional_landmark', 's', '"the appliance you cook on" *'), ('positive_contrast', '', 'distractor mentioned'),
         ('negative_contrast', '', 'target + "not the one on …"'), ('self_correction', 's', 'distractor first, "sorry, I mean …" *'),
         ('target_cue_region', '', 'table-region cue'), ('code_switch', 's', '"on the 炉子" *'),
         ('translate_zh', 's', 'full Chinese *'), ('negation_only', 's', '"the bowl that is not on …" *')]
fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
ys = np.arange(len(items))[::-1]
for y, (c, scr, lab) in zip(ys, items):
    for dy, (note, colr, name) in zip((0.17, -0.17), ((P, '#2a78d6', 'π0.5-LIBERO'), (OV, '#eb6834', 'OpenVLA official'))):
        p = per(c, note + ('--screen' if scr else ''))
        k = p[7][0] + p[9][0]; n = p[7][1] + p[9][1]; lo, hi = wil(k, n)
        ax.plot([lo * 100, hi * 100], [y + dy, y + dy], color=colr, alpha=0.35, lw=2, solid_capstyle='round', zorder=1)
        ax.scatter([100 * k / n], [y + dy], s=40, color=colr, zorder=2, edgecolors='#ffffff', linewidths=1.5,
                   label=name if y == ys[0] else None)
ax.set_yticks(ys); ax.set_yticklabels([i[2] for i in items], fontsize=8.5, color='#2b2a27')
ax.set_xlim(0, 100); ax.set_xlabel('success rate on tasks 7 + 9 (%), 95% Wilson CI; * = 10 rollouts', fontsize=8.5, color='#4a4945')
for sp in ('top', 'right', 'left'): ax.spines[sp].set_visible(False)
ax.spines['bottom'].set_color('#c9c7c0'); ax.tick_params(axis='y', length=0); ax.tick_params(axis='x', colors='#6b6a64', labelsize=8)
ax.grid(axis='x', color='#ecebe7', lw=0.8); ax.set_axisbelow(True)
ax.legend(loc='lower right', frameon=False, fontsize=8.5)
fig.text(0.02, 0.95, 'Mirror pair (tasks 7/9): π0.5 vs. OpenVLA when only language can pick the bowl', fontsize=11.5, color='#2b2a27')
plt.subplots_adjust(left=0.36, right=0.97, top=0.91, bottom=0.08)
plt.savefig(os.path.join(OUT, 'mirror_pair_pi05_vs_openvla.png')); plt.close()
