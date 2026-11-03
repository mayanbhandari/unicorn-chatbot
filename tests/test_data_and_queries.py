# Tests for the data cleaning and the pandas filters, using the real CSV. We need them to be sure the numbers behind the bot's lists and rankings are right, since those answers skip the LLM.

from src.data_loader import SECTOR_GROUPS, parse_employees, parse_money_usd_m
from src.structured_query import Filters, apply_filters, find_companies


def test_dataset_loaded_and_enriched(df):
    # all 118 companies came through
    assert len(df) == 118
    assert (df["sector_group"] != "Other").all(), "every primary_sector must be mapped to a group"
    # no city name should still have a trailing space
    assert df["location"].str.endswith(" ").sum() == 0
    # the rupee sign was restored (the raw file had "?")
    assert any("₹" in v for v in df["annual_revenue"].dropna())


# guards against a new sector appearing in the data without a group
def test_group_mapping_covers_all_sectors(df):
    mapped = {s for items in SECTOR_GROUPS.values() for s in items}
    assert set(df["primary_sector"]) <= mapped


def test_parsers():
    # $5B is 5000 million dollars
    assert parse_money_usd_m("$5B as on Aug 14, 2024") == 5000
    assert parse_money_usd_m("$502M") == 502
    # rupees are converted at the fixed rate, so we only check a sensible range
    assert 1100 < parse_money_usd_m("₹10,100Cr as on Sep 17, 2024") < 1250
    # missing values should give None, not crash
    assert parse_money_usd_m(None) is None
    assert parse_employees("2,368 as on Feb 28, 2025") == 2368


# lowercase "fintech" should still match the "Fintech" group
def test_fintech_filter(df):
    r = apply_filters(df, Filters(sector_groups=["fintech"]))
    assert len(r) == 29 and "Razorpay" in set(r["company"])


def test_primary_sector_keywords_treat_regex_symbols_literally(df):
    # A dot is a regex wildcard unless escaped; literally no sector name contains a dot.
    assert apply_filters(df, Filters(primary_sector_keywords=["."])).empty


# "Bangalore" must map to "Bengaluru", and searching inside a previous list must only narrow it
def test_bangalore_alias_and_followup_restriction(df):
    fintech = apply_filters(df, Filters(sector_groups=["Fintech"]))
    blr = apply_filters(df, Filters(locations=["Bangalore"]), candidate_ids=fintech["id"].tolist())
    assert 0 < len(blr) < len(fintech) and set(blr["location"]) == {"Bengaluru"}


# "the most valuable company" = sort by valuation, keep the top 1
def test_sort_and_limit(df):
    top = apply_filters(df, Filters(sort_by="valuation", limit=1))
    assert top.iloc[0]["company"] == "Flipkart"


def test_founded_year_bounds_are_strict(df):
    # In ordinary wording, "after 2018" and "before 2018" exclude the boundary year.
    after = apply_filters(df, Filters(founded_after=2018))
    before = apply_filters(df, Filters(founded_before=2018))
    assert (after["founded_year"] > 2018).all()
    assert (before["founded_year"] < 2018).all()


# a typo ("razor pay") still finds Razorpay, and a made-up name is reported as missing
def test_fuzzy_company_lookup(df):
    rows, missing = find_companies(df, ["razor pay", "Zepto", "NotARealCo"])
    assert set(rows["company"]) == {"Razorpay", "Zepto"} and missing == ["NotARealCo"]
