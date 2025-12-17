def tax_benefits(schedule, tax_rate):
    yearly = {}

    for item in schedule:
        year = (item["month"] - 1) // 12 + 1
        yearly.setdefault(year, {"interest": 0, "principal": 0})
        yearly[year]["interest"] += item["interest"]
        yearly[year]["principal"] += item["principal"]

    total = 0
    for year, data in yearly.items():
        interest_deduction = min(200000, data["interest"])
        principal_deduction = min(150000, data["principal"])
        deduction = interest_deduction + principal_deduction

        total += deduction * (tax_rate / 100)

    return total
