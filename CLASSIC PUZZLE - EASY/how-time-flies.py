import sys
import math
import datetime
from dateutil import relativedelta

def format_string(m, y, total_days):
    s = ''
    if y == 1:
        s += '1 year, '
    elif y > 1:
        s += '{} years, '.format(y)
    if m == 1:
        s += '1 month, '
    elif m > 1:
        s += '{} months, '.format(m)
    s += 'total {} days'.format(total_days)
    return s

d, m, y = list(map(int, input().split('.')))
begin = datetime.date(day=d, month=m, year=y)
d, m, y = list(map(int, input().split('.')))
end = datetime.date(day=d, month=m, year=y)
delta = relativedelta.relativedelta(end, begin)
print(format_string(delta.months, delta.years, (end - begin).days))
