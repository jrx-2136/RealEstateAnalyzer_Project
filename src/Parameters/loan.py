def calculate_emi(principal, annual_rate, tenure_years=20):
    r = annual_rate / 12 / 100
    n = tenure_years * 12
    emi = principal * r * (1 + r)**n / ((1 + r)**n - 1)
    return emi


def amortization_schedule(principal, annual_rate, tenure_years=20):
    emi = calculate_emi(principal, annual_rate, tenure_years)
    r = annual_rate / 12 / 100
    balance = principal
    schedule = []

    for month in range(1, tenure_years * 12 + 1):
        interest = balance * r
        principal_paid = emi - interest
        balance -= principal_paid

        schedule.append({
            "month": month,
            "emi": emi,
            "interest": interest,
            "principal": principal_paid,
            "balance": max(balance, 0)
        })

    return schedule
