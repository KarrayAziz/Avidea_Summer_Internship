#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

VALID_EXTENSIONS={'.jpg','.jpeg','.png','.bmp','.webp'}

class RobustnessDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows=rows; self.transform=transform
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r=self.rows[i]
        img=Image.open(r['candidate_path']).convert('RGB')
        return self.transform(img), r['candidate_path'], r['severity'], r['crop_type']

def build_model(dropout):
    m=models.mobilenet_v3_small(weights=None)
    in_features=m.classifier[3].in_features
    m.classifier[2]=nn.Dropout(p=dropout,inplace=True)
    m.classifier[3]=nn.Linear(in_features,2)
    return m

def build_transform(size):
    return transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

def norm(s): return s.strip().lower().replace(' ','_').replace('-','_')

def find_col(fields, names):
    mp={norm(x):x for x in fields}
    for n in names:
        if norm(n) in mp: return mp[norm(n)]
    return None

def resolve_path(raw, manifest, candroot):
    p=Path(raw).expanduser()
    for q in [p, manifest.parent/p, candroot/p, candroot/p.name]:
        if q.exists() and q.is_file(): return q.resolve()
    matches=list(candroot.rglob(p.name))
    return matches[0].resolve() if len(matches)==1 else None

def load_rows(root, severity):
    manifest=root/severity/'manifest.csv'
    candroot=root/severity/'candidates'
    with manifest.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f); fields=reader.fieldnames or []; rows=list(reader)
    crop_col=find_col(fields,['crop_type','crop_direction','direction','crop'])
    path_col=find_col(fields,['candidate_path','output_path','generated_path','crop_path','image_path','path','candidate','output','generated_image','filename','file_name'])
    if not crop_col: raise RuntimeError(f'No crop type column in {manifest}: {fields}')
    out=[]
    if path_col:
        for r in rows:
            p=resolve_path(str(r[path_col]),manifest,candroot)
            if p: out.append({'candidate_path':str(p),'severity':severity,'crop_type':str(r[crop_col])})
    else:
        files=sorted(p for p in candroot.rglob('*') if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS)
        if len(files)!=len(rows):
            raise RuntimeError(f'Cannot pair manifest rows and files for {severity}: {len(rows)} vs {len(files)}')
        for r,p in zip(rows,files):
            out.append({'candidate_path':str(p.resolve()),'severity':severity,'crop_type':str(r[crop_col])})
    return out

def summarize(rows):
    probs=[float(r['probability_incomplete']) for r in rows]
    preds=[bool(r['predicted_incomplete']) for r in rows]
    n=len(rows); det=sum(preds)
    return {
        'count':n,
        'detected_incomplete':det,
        'missed_incomplete':n-det,
        'incomplete_recall':det/n if n else 0.0,
        'mean_probability_incomplete':float(np.mean(probs)) if probs else 0.0,
        'median_probability_incomplete':float(np.median(probs)) if probs else 0.0,
    }

def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def main(a):
    root=Path(a.robustness_root).expanduser().resolve()
    ckpt_path=Path(a.checkpoint).expanduser().resolve()
    outdir=Path(a.output).expanduser().resolve(); outdir.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if a.device=='auto' and torch.cuda.is_available() else a.device)
    ckpt=torch.load(ckpt_path,map_location=device)
    size=int(ckpt.get('image_size',224)); dropout=float(ckpt.get('dropout',0.3))
    model=build_model(dropout); model.load_state_dict(ckpt['model_state_dict']); model.to(device).eval()
    if a.threshold is None:
        with (ckpt_path.parent/'selected_threshold.json').open() as f: threshold=float(json.load(f)['selected_threshold'])
    else: threshold=float(a.threshold)
    rows=[]
    for sev in a.severities: rows.extend(load_rows(root,sev))
    loader=DataLoader(RobustnessDataset(rows,build_transform(size)),batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,pin_memory=device.type=='cuda')
    preds=[]
    with torch.no_grad():
        for images,paths,sevs,crops in tqdm(loader,desc='Evaluating'):
            images=images.to(device,non_blocking=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=(a.amp and device.type=='cuda')):
                probs=torch.softmax(model(images),dim=1)[:,1].cpu().tolist()
            for p,s,c,pr in zip(paths,sevs,crops,probs):
                pred=pr>=threshold
                preds.append({'path':p,'severity':s,'crop_type':c,'probability_incomplete':float(pr),'threshold':threshold,'predicted_incomplete':pred,'correct':pred})
    sev_groups=defaultdict(list); crop_groups=defaultdict(list)
    for r in preds:
        sev_groups[r['severity']].append(r); crop_groups[(r['severity'],r['crop_type'])].append(r)
    sev_summary=[{'severity':s,**summarize(sev_groups[s])} for s in a.severities]
    crop_summary=[{'severity':s,'crop_type':c,**summarize(rs)} for (s,c),rs in sorted(crop_groups.items())]
    write_csv(outdir/'robustness_predictions.csv',preds)
    write_csv(outdir/'summary_by_severity.csv',sev_summary)
    write_csv(outdir/'summary_by_crop_type.csv',crop_summary)
    with (outdir/'robustness_summary.json').open('w') as f: json.dump({'overall':summarize(preds),'by_severity':sev_summary,'by_crop_type':crop_summary},f,indent=2)
    print('\nBY SEVERITY')
    for r in sev_summary:
        print(f"{r['severity']:9} n={r['count']:4d} recall={r['incomplete_recall']:.4f} mean_P={r['mean_probability_incomplete']:.4f}")
    print('\nBY SEVERITY AND CROP TYPE')
    for r in crop_summary:
        print(f"{r['severity']:9} {r['crop_type']:13} n={r['count']:3d} recall={r['incomplete_recall']:.4f} mean_P={r['mean_probability_incomplete']:.4f}")

def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--robustness-root',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',default='./back_robustness_results')
    p.add_argument('--severities',nargs='+',default=['subtle','moderate','strong'])
    p.add_argument('--threshold',type=float,default=None)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--num-workers',type=int,default=4)
    p.add_argument('--device',default='auto')
    p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True)
    return p

if __name__=='__main__': main(parser().parse_args())
