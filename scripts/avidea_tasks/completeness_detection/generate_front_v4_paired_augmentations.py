#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, random, shutil
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

EXTS={'.jpg','.jpeg','.png','.bmp','.webp'}
VEHICLE_CLASSES=[2,5,7]

def args():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--yolo',type=Path,default=Path('/home/aziz/Aziz/DigiCover/usingGeminiApi/models/yolov8m.pt'))
    p.add_argument('--view',default='front')
    p.add_argument('--complete-count',type=int,default=40)
    p.add_argument('--incomplete-count',type=int,default=80)
    p.add_argument('--conf',type=float,default=0.30)
    p.add_argument('--min-area-ratio',type=float,default=0.08)
    p.add_argument('--min-output-size',type=int,default=224)
    p.add_argument('--device',default='0')
    p.add_argument('--seed',type=int,default=1604)
    p.add_argument('--save-previews',action='store_true')
    p.add_argument('--overwrite',action='store_true')
    return p.parse_args()

def images(folder):
    out=[]
    for p in folder.rglob('*'):
        if p.is_file() and p.suffix.lower() in EXTS:
            parts=[x.lower() for x in p.relative_to(folder).parts]
            if not any('synthetic' in x for x in parts): out.append(p)
    return sorted(out)

def largest_box(model,img,a):
    h,w=img.shape[:2]; area=h*w; best=None; ba=-1
    rs=model.predict(source=img,conf=a.conf,classes=VEHICLE_CLASSES,device=a.device,verbose=False)
    if not rs or rs[0].boxes is None: return None
    for b in rs[0].boxes.xyxy.detach().cpu().numpy():
        x1,y1,x2,y2=map(float,b); ar=max(0,x2-x1)*max(0,y2-y1)
        if ar/area<a.min_area_ratio: continue
        if ar>ba:
            ba=ar; best=(max(0,int(np.floor(x1))),max(0,int(np.floor(y1))),min(w,int(np.ceil(x2))),min(h,int(np.ceil(y2))))
    return best

def hard_complete(shape,b,rng,minsize):
    h,w=shape[:2]; x1,y1,x2,y2=b; bw=x2-x1; bh=y2-y1
    kind=rng.choices(['tight_top','tight_right','tight_bottom','tight_left','tight_all'],weights=[40,20,20,10,10],k=1)[0]
    rx=lambda:max(6,round(bw*rng.uniform(.025,.06))); tx=lambda:max(3,round(bw*rng.uniform(.008,.025)))
    ry=lambda:max(6,round(bh*rng.uniform(.03,.08))); ty=lambda:max(3,round(bh*rng.uniform(.008,.025)))
    ml,mr,mt,mb=rx(),rx(),ry(),ry()
    if kind=='tight_top': mt=ty()
    elif kind=='tight_right': mr=tx()
    elif kind=='tight_bottom': mb=ty()
    elif kind=='tight_left': ml=tx()
    else: ml,mr,mt,mb=tx(),tx(),ty(),ty()
    c=(max(0,x1-ml),max(0,y1-mt),min(w,x2+mr),min(h,y2+mb))
    cx1,cy1,cx2,cy2=c
    if cx2-cx1<minsize or cy2-cy1<minsize:return None
    if not(cx1<=x1 and cy1<=y1 and cx2>=x2 and cy2>=y2):return None
    return c,kind

def severity(rng):
    r=rng.random()
    if r<.80:return rng.uniform(.01,.035),'01_035'
    if r<.97:return rng.uniform(.035,.06),'035_06'
    return rng.uniform(.06,.09),'06_09'

def incomplete(shape,b,rng,minsize):
    h,w=shape[:2]; x1,y1,x2,y2=b; bw=x2-x1; bh=y2-y1
    kind=rng.choices(['top','right','bottom','left'],weights=[50,20,20,10],k=1)[0]
    s,band=severity(rng); cx1=0;cy1=0;cx2=w;cy2=h
    if kind=='top': cy1=max(0,min(h-1,y1+max(2,round(bh*s))))
    elif kind=='right': cx2=max(1,min(w,x2-max(2,round(bw*s))))
    elif kind=='bottom': cy2=max(1,min(h,y2-max(2,round(bh*s))))
    else: cx1=max(0,min(w-1,x1+max(2,round(bw*s))))
    if cx2-cx1<minsize or cy2-cy1<minsize:return None
    return (cx1,cy1,cx2,cy2),kind,s,band

def preview(img,b,c,path):
    im=img.copy(); x1,y1,x2,y2=b; a1,b1,a2,b2=c
    cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,0),3)
    cv2.rectangle(im,(a1,b1),(a2,b2),(0,0,255),3)
    path.parent.mkdir(parents=True,exist_ok=True); cv2.imwrite(str(path),im)

def main():
    a=args(); rng=random.Random(a.seed); src=a.dataset/'train'/a.view/'complete'
    if a.overwrite and a.output.exists():shutil.rmtree(a.output)
    cdir=a.output/'candidates'/'complete'/a.view; idir=a.output/'candidates'/'incomplete'/a.view
    cpdir=a.output/'previews'/'complete'/a.view; ipdir=a.output/'previews'/'incomplete'/a.view
    cdir.mkdir(parents=True,exist_ok=True); idir.mkdir(parents=True,exist_ok=True)
    model=YOLO(str(a.yolo)); det=[]
    for p in images(src):
        im=cv2.imread(str(p))
        if im is None:continue
        b=largest_box(model,im,a)
        if b:det.append((p,im,b))
    rows=[]
    for label,count,fn,od,pd in [('complete',a.complete_count,hard_complete,cdir,cpdir),('incomplete',a.incomplete_count,incomplete,idir,ipdir)]:
        made=0;attempt=0
        while made<count and attempt<max(1600,count*100):
            p,im,b=det[attempt%len(det)];attempt+=1
            r=fn(im.shape,b,rng,a.min_output_size)
            if not r:continue
            if label=='complete': c,kind=r; s='';band=''
            else: c,kind,s,band=r
            x1,y1,x2,y2=c; crop=im[y1:y2,x1:x2]
            name=f'{made+1:04d}__{p.stem}__{label}__{kind}__{band}.jpg'
            out=od/name
            if not cv2.imwrite(str(out),crop):continue
            if a.save_previews:preview(im,b,c,pd/name)
            rows.append({'label':label,'source_path':str(p),'output_path':str(out),'crop_type':kind,'severity':s,'severity_band':band,'vehicle_box':','.join(map(str,b)),'crop_box':','.join(map(str,c))})
            made+=1
        print(f'{label}: {made}/{count}')
    with (a.output/'manifest.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print('Output:',a.output)

if __name__=='__main__':main()
