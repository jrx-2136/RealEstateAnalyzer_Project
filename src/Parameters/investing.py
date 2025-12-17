def future_value_lumpsum(P, rate, years):
    return P * (1 + rate / 100)**years


def future_value_sip(monthly, rate, years):
    r = rate / 12 / 100
    n = years * 12
    return monthly * ((1 + r)**n - 1) / r
