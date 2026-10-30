"""Bounded local recovery from repeated immediate collision rejections.

This is evidence of a poor search starting point, not proof of inevitability.
It changes checkpoint choice only, never physics limits or sampling ranges.
"""
import json
from pathlib import Path


def recommend(ledger, load_rows):
    if not ledger or not ledger[-1].get('failure'):
        return None
    end=float(ledger[-1]['to_s'])
    # Previously accepted later motion invalidates an old-frontier diagnosis.
    if any(float(row.get('to_s',0.)) > end+1e-6 for row in ledger):
        return None
    qualifying=[]
    for row in reversed(ledger[-12:]):
        if abs(float(row.get('to_s',0.))-end)>1e-6 or not row.get('failure'):
            continue
        samples=load_rows(row['run'],end)
        if len(samples)<4:
            continue
        if all(not x.get('ok') and int(x.get('step',999))<=3
               and 'cross wrist clearance' in str(x.get('reason','')) for x in samples):
            qualifying.append(row['run'])
    if not qualifying:
        return None
    distance=.2*min(3,len(qualifying))
    return dict(reason='Repeated shortlisted candidates fail cross-wrist clearance within 20ms',
                blocked_time_s=end, qualifying_runs=qualifying, max_recovery_time_s=round(end-distance,8),
                max_backoff_s=.6, evidence_scope='shortlisted candidates, not all possible controls')


def from_history(root, ledger):
    root=Path(root)
    def load(run,end):
        path=root/'physics'/run
        # Compare failures under the same continuous-guidance family only.
        if not (path/'clearance_guidance_metrics.json').exists():
            return []
        file=path/'shortlist_audit.jsonl'
        if not file.exists():return []
        with file.open('rb') as stream:
            stream.seek(max(0,file.stat().st_size-131072));data=stream.read().decode(errors='replace')
        rows=[]
        for line in data.splitlines():
            try:row=json.loads(line)
            except ValueError:continue
            if abs(float(row.get('time_s',-1))-end)<1e-6:rows.append(row)
        return rows[-12:]
    return recommend(ledger,load)
