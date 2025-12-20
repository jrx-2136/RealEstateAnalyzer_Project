def renting_case(
    initial_rent,
    escalation,
    down_payment,
    invest_rate,
    monthly_saving,
    tenure_years=20
):
    months = tenure_years * 12
    annual_rate = invest_rate / 100
    monthly_rate = annual_rate / 12

    # -----------------------------
    # Total Rent Paid (with escalation)
    # -----------------------------
    total_rent_paid = 0
    current_rent = initial_rent

    for year in range(tenure_years):
        total_rent_paid += current_rent * 12
        current_rent *= (1 + escalation / 100)

    # -----------------------------
    # Lump Sum Investment (Down Payment)
    # -----------------------------
    lump_sum_value = down_payment * ((1 + annual_rate) ** tenure_years)

    # -----------------------------
    # SIP Investment (Monthly Savings)
    # -----------------------------
    sip_value = monthly_saving * (
        ((1 + monthly_rate) ** months - 1) / monthly_rate
    ) * (1 + monthly_rate)

    # -----------------------------
    # Final Rental Wealth
    # -----------------------------
    final_wealth_renting = lump_sum_value + sip_value

    return {
        "total_rent_paid": round(total_rent_paid, 2),
        "lump_sum_value": round(lump_sum_value, 2),
        "sip_value": round(sip_value, 2),
        "wealth_renting": round(final_wealth_renting, 2)
    }
def compare_results(buy, rent):
    if buy["wealth_buying"] > rent["wealth_renting"]:
        return "BUYING is financially better"
    elif rent["wealth_renting"] > buy["wealth_buying"]:
        return "RENTING is financially better"
    else:
        return "Both options yield similar financial outcome"
