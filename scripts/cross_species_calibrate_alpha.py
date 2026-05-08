import argparse, json

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out_json', required=True); p.add_argument('--pairs', nargs='+', required=True, help='e.g. ARID1A=-2.0 PDCD1=-0.25')
    a=p.parse_args(); d={}
    for item in a.pairs:
        k,v=item.split('='); d[k]=float(v)
    with open(a.out_json,'w') as f: json.dump(d,f,indent=2,sort_keys=True)
    print('saved',a.out_json)
if __name__=='__main__': main()
