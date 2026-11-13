"""现有业务摘要，不推断未观察到的业务关系。"""
def analyze(snapshot):
    variables=snapshot.metadata['variables'];lags=snapshot.metadata['lags'];rows=snapshot.rows('months');by_month={}
    for r in rows:
        if r['month'] in by_month:raise ValueError('月份重复')
        by_month[r['month']]=r['values']
    result=[]
    for month in sorted(by_month):
        window=[by_month.get(month-j) for j in range(lags+1)]
        if any(w is None or any(w.get(v) is None for v in variables) for w in window):continue
        result.append({'month':month,'Y':[window[0][v] for v in variables],
                       'X':[1]+[window[j][v] for j in range(1,lags+1) for v in variables]})
    return result
