"""Conditional-state diagnostics and a 2021 ECB AAA forward-discount proxy.

Reads licensed prices locally; exports aggregate statistics, never daily prices.
The ECB curve is a sovereign-bond proxy, not an OIS or a calibrated banking factor.
"""
from pathlib import Path
from datetime import date, datetime, timezone
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor
import csv, hashlib, json, math
from openpyxl import load_workbook

from release_paths import results_dir, contract_path
ROOT = results_dir()/'estimation'
CACHE = results_dir()/'ecb_cache'
KEYS = ['BETA0','BETA1','BETA2','BETA3','TAU1','TAU2','SR_1Y','SR_2Y']
PREFIX = 'https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.'
QUERY = '?startPeriod=2021-01-01&endPeriod=2021-12-31'
T1, T2 = date(2022,12,19), date(2023,12,18)
PI1, PI2 = 121.01,128.71

def read_contract(year):
    file = contract_path(year)
    wb = load_workbook(file,read_only=True,data_only=True)
    out = {r[0].date():float(r[1]) for r in wb.active.iter_rows(values_only=True)
           if isinstance(r[0],datetime) and isinstance(r[1],(int,float))
           and r[0].year == 2021 and math.isfinite(r[1])}
    wb.close()
    return out, {'file':file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}

def load_curve(key):
    file = CACHE / f'ECB_YC_AAA_2021_{key}.csv'
    url = PREFIX+key+QUERY
    if not file.exists():
        error = None
        for attempt in range(2):
            try:
                req=Request(url,headers={'Accept':'text/csv','User-Agent':'AcademicStateDomainAudit/1.0'})
                with urlopen(req,timeout=25) as response:
                    raw=response.read()
                if b'TIME_PERIOD' not in raw or b'OBS_VALUE' not in raw:
                    raise ValueError('API response has no expected CSV header')
                CACHE.mkdir(parents=True,exist_ok=True)
                file.write_bytes(raw)
                error=None
                break
            except Exception as exc:
                error=exc
        if error is not None:
            return key,{}, {'url':url,'error':str(error)}
    with file.open(encoding='utf-8-sig',newline='') as stream:
        rows={date.fromisoformat(r['TIME_PERIOD']):float(r['OBS_VALUE'])
              for r in csv.DictReader(stream) if r.get('TIME_PERIOD') and r.get('OBS_VALUE')}
    return key, rows, {'url':url,'file':file.name,'observations':len(rows),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}

def zero_pct(t,parameters):
    x1=t/parameters['TAU1'];x2=t/parameters['TAU2']
    l1=-math.expm1(-x1)/x1;l2=-math.expm1(-x2)/x2
    return parameters['BETA0']+parameters['BETA1']*l1+parameters['BETA2']*(l1-math.exp(-x1))+parameters['BETA3']*(l2-math.exp(-x2))

def stats(values):
    return {'min':min(values),'max':max(values),'mean':sum(values)/len(values)}

def domain(a,b,days,kappas):
    gaps=[a[d]-kappas[d]*b[d] for d in days]
    return {'observations':len(days),'X1_positive_n':sum(x>0 for x in gaps),
            'X1_nonpositive_n':sum(x<=0 for x in gaps),'X1_ge_pi1_n':sum(x>=PI1 for x in gaps),
            'X2_outside_0_pi2_n':sum(not 0<b[d]<PI2 for d in days),
            'both_factors_legal_n':sum(0<x<PI1 and 0<b[d]<PI2 for x,d in zip(gaps,days)),
            'X1_eur_range':[min(gaps),max(gaps)],'X2_eur_range':[min(b[d] for d in days),max(b[d] for d in days)]}

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    a, ma=read_contract(22);b,mb=read_contract(23)
    days=sorted(set(a)&set(b))
    ratios={d:a[d]/b[d] for d in days}
    critical_day=min(ratios,key=ratios.get);critical_k=ratios[critical_day]
    gap=(T2-T1).days/365.25
    report={'generated_utc':datetime.now(timezone.utc).isoformat(),'price_files':[ma,mb],
            'sample':{'start':str(days[0]),'end':str(days[-1]),'n':len(days)},
            'proxy_dates':{'T1':str(T1),'T2':str(T2),'gap_ACT36525':gap,
                           'basis':'Last observation dates in source workbooks; not verified compliance dates'},
            'penalty_inputs':{'pi1':PI1,'pi2':PI2},
            'algebraic_threshold':{'constant_kappa_strict_upper_bound':critical_k,
                                   'binding_day':str(critical_day),'flat_one_year_rate_strict_lower_bound_pct':-100*math.log(critical_k),
                                   'flat_proxy_date_gap_rate_strict_lower_bound_pct':-100*math.log(critical_k)/gap,
                                   'interpretation':'Algebraic positivity condition only, not an estimated or justified interest rate'},
            'old_3pct_condition':{'kappa':math.exp(-.03),**domain(a,b,days,{d:math.exp(-.03) for d in days})},
            'unit_kappa_condition':{'kappa':1.,**domain(a,b,days,{d:1. for d in days})}}
    with ThreadPoolExecutor(max_workers=4) as pool:
        fetched=list(pool.map(load_curve,KEYS))
    series={k:s for k,s,m in fetched};report['ecb_sources']={k:m for k,s,m in fetched}
    if all(series[k] for k in KEYS):
        all_curve_dates=sorted(set.intersection(*(set(s) for s in series.values())))
        matched=sorted(set(days)&set(all_curve_dates))
        kappas={};forwards=[];reproduction=[]
        for d in matched:
            pars={k:series[k][d] for k in KEYS[:6]}
            u1=(T1-d).days/365.25;u2=(T2-d).days/365.25
            integrated=zero_pct(u2,pars)*u2-zero_pct(u1,pars)*u1
            kappas[d]=math.exp(-integrated/100.)
            forwards.append(integrated/gap)
            reproduction.extend([abs(zero_pct(1.,pars)-series['SR_1Y'][d]),abs(zero_pct(2.,pars)-series['SR_2Y'][d])])
        report['ecb_AAA_forward_proxy']={'kind':'ECB AAA nominal sovereign zero curve; not OIS',
                'method':'P_t(T2)/P_t(T1), Svensson continuously compounded zero yields, ACT/365.25',
                'curve_dates_n':len(all_curve_dates),'matched_dates_n':len(matched),
                'unmatched_price_dates':[str(d) for d in days if d not in matched],
                'kappa':stats(list(kappas.values())),'annual_forward_rate_pct':stats(forwards),
                'max_zero_yield_1y_2y_reproduction_error_pct_points':max(reproduction),
                **domain(a,b,matched,kappas)}
    else:
        report['ecb_AAA_forward_proxy']={'error':'One or more API series could not be loaded; no proxy state diagnostic claimed'}
    output=ROOT/'external_kappa_audit_summary.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
