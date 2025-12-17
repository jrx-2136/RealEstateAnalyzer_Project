from .loan import amortization_schedule
from .tax import tax_benefits
from .investing import future_value_lumpsum, future_value_sip


def property_appreciation(value, rate, years=20):
    return value * (1 + rate / 100)**years


def buying_case(price, down_payment, loan_rate, tax_rate, appreciation_rate):
    principal = price - down_payment
    schedule = amortization_schedule(principal, loan_rate)

    total_emi = sum(item["emi"] for item in schedule)
    tax_saved = tax_benefits(schedule, tax_rate)
    net_emi = total_emi - tax_saved
    final_asset = property_appreciation(price, appreciation_rate)

    return {
        "total_emi": total_emi,
        "tax_saved": tax_saved,
        "net_emi": net_emi,
        "final_asset": final_asset,
        "wealth_buying": final_asset - net_emi
    }


def renting_case(rent_start, escalation, down_payment, invest_rate, monthly_saving, years=20):
    rent = rent_start
    total_rent = 0

    for _ in range(years):
        total_rent += rent * 12
        rent *= (1 + escalation / 100)

    lump_future = future_value_lumpsum(down_payment, invest_rate, years)
    sip_future = future_value_sip(monthly_saving, invest_rate, years)

    return {
        "total_rent": total_rent,
        "lump_future": lump_future,
        "sip_future": sip_future,
        "wealth_renting": lump_future + sip_future - total_rent
    }


def compare_results(buy, rent):
    if buy["wealth_buying"] > rent["wealth_renting"]:
        return "BUYING is better"
    return "RENTING is better"
