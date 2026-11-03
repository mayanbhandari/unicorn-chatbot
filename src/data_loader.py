#Loads the Tracxn CSV and cleans it up: fixes the encoding, adds sector groups, turns valuation and headcount text into numbers. We need it because the raw file is messy, and the rest of the bot assumes clean, consistent data.

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# The raw data has 56 very specific sectors (Payments, Alternative Lending...), but nobody asks for those.
# People say "fintech" or "edtech", so I grouped them by hand into ten buckets.
SECTOR_GROUPS: dict[str, list[str]] = {
    "Fintech": [
        "Payments", "Alternative Lending", "Banking Tech", "Investment Tech",
        "Internet First Insurance Platforms", "Cryptocurrencies",
        "Consumer and SME Loans", "Finance & Accounting Tech",
    ],
    "E-commerce & Consumer Brands": [
        "Horizontal E-Commerce", "B2B E-Commerce", "E-Commerce Enablers",
        "Consumer Electronics E-Commerce", "Social Commerce", "Fashion Tech",
        "Brands Tech", "Beauty Tech", "Beauty & Personal Care Products",
        "Mom & Baby Care", "Online Grocery",
    ],
    "Logistics & Mobility": [
        "Logistics Tech", "Road Transport Tech", "Electric Vehicles",
        "Auto E-Commerce & Content",
    ],
    "EdTech": ["K-12 EdTech", "Continued Learning", "Test Preparation Tech"],
    "HealthTech": [
        "Healthcare Booking Platforms", "Infectious Diseases", "Employee Health IT",
        "Healthcare Payer Tech", "Healthcare IT",
    ],
    "Enterprise SaaS & Dev Tools": [
        "Sales Force Automation", "HRTech", "API Management", "Business Intelligence",
        "Software Testing Tools", "SME Business Management Software",
        "Customer Service Software", "Legal Tech", "Cybersecurity", "AdTech and Marketing",
    ],
    "AI & Deep Tech": ["Generative AI", "Speech and Voice Recognition", "Chatbots"],
    "Media, Gaming & Social": [
        "Vernacular Platforms", "PC & Console Gaming", "Mobile Gaming", "Sports Tech",
        "Tech for Traditional Advertising", "Mobile Utility Tools",
    ],
    "Consumer Services & Travel": ["Food Tech", "Online Travel", "Local Services"],
    "Real Estate & Construction": [
        "Residential Real Estate Tech", "Construction Tech", "Home Improvements",
    ],
}
# Flip it around so we can look up a group from a sector name (lowercase, so case doesn't matter).
_SECTOR_TO_GROUP = {s.lower(): g for g, items in SECTOR_GROUPS.items() for s in items}

# Users say "Bangalore", the data says "Bengaluru". These maps translate what people type
# into what is actually in the file. One alias can point to several cities (NCR).
LOCATION_ALIASES: dict[str, list[str]] = {
    "bangalore": ["Bengaluru"], "bengaluru": ["Bengaluru"], "bengalore": ["Bengaluru"],
    "gurgaon": ["Gurugram"], "gurugram": ["Gurugram"],
    "bombay": ["Mumbai"], "mumbai": ["Mumbai"],
    "new delhi": ["Delhi"], "delhi": ["Delhi"],
    "ncr": ["Delhi", "Gurugram", "Noida"], "delhi ncr": ["Delhi", "Gurugram", "Noida"],
    "madras": ["Chennai"], "bay area": ["San Francisco", "Palo Alto", "Mountain View", "Santa Clara", "San Mateo"],
    "nyc": ["New York City"], "new york": ["New York City"],
}
# Same idea for countries, mainly so "US" matches "United States".
COUNTRY_ALIASES = {"us": "United States", "usa": "United States", "u.s.": "United States",
                   "america": "United States", "united states of america": "United States"}


def normalise_locations(values: list[str]) -> list[str]:
    out: list[str] = []
    for v in values:
        key = v.strip().lower()
        # use the alias if we know one, otherwise keep what the user typed
        out.extend(LOCATION_ALIASES.get(key, [v.strip()]))
    # dict.fromkeys drops duplicates but keeps the order
    return list(dict.fromkeys(out))


def normalise_country(value: str) -> str:
    return COUNTRY_ALIASES.get(value.strip().lower(), value.strip())


# Regex piece for a number like 1,04,000 or 4.76 (commas allowed, decimals optional).
_NUM = r"([\d,]+(?:\.\d+)?)"


def parse_money_usd_m(text: str | float | None, inr_per_usd: float = 85.0) -> float | None:
    """'$5B as on ...' -> 5000.0 ; '₹10,100Cr' -> ~1188 ; '$4.76B - $7.97B' -> midpoint. Returns USD millions."""
    # empty or missing value: nothing to parse
    if not isinstance(text, str) or not text.strip():
        return None
    values: list[float] = []
    # find every money amount in the string (a range like "$4.76B - $7.97B" has two)
    for m in re.finditer(r"([$₹])\s*" + _NUM + r"\s*(B|M|K|Cr|L)?", text):
        cur, num, unit = m.group(1), float(m.group(2).replace(",", "")), m.group(3)
        if cur == "$":
            # B = billion, M = million, K = thousand; no suffix means plain dollars, so scale it way down
            mult = {"B": 1000.0, "M": 1.0, "K": 0.001}.get(unit or "", 1e-6)
            values.append(num * mult)
        else:  # INR: Cr = 1e7 rupees, L = 1e5 rupees
            rupees = num * {"Cr": 1e7, "L": 1e5}.get(unit or "", 1.0)
            # rupees -> dollars -> millions of dollars
            values.append(rupees / inr_per_usd / 1e6)
    if not values:
        return None
    # for a range we just take the midpoint; good enough since this number is only used for sorting
    return round(sum(values) / len(values), 2)


def parse_employees(text: str | float | None) -> int | None:
    if not isinstance(text, str):
        return None
    # values look like "2,368 as on Feb 28, 2025", so grab the number at the start
    m = re.match(r"\s*([\d,]+)", text)
    return int(m.group(1).replace(",", "")) if m else None


def parse_unicorn_year(text: str) -> int | None:
    # the company_info text says "joined the Unicorn Club on Apr 07, 2025"; pull out just the year
    m = re.search(r"joined the Unicorn Club on \w+ \d{1,2}, (\d{4})", text or "")
    return int(m.group(1)) if m else None


def _read_csv(path: Path) -> pd.DataFrame:
    # The file isn't plain UTF-8 (it has stray non-breaking spaces), so try a few encodings until one works.
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path}")


def _fix_rupee(value):
    # The exporter turned the rupee sign into '?': '?348Cr' -> '₹348Cr'
    # The lookahead only matches a '?' that is followed by a digit, so real question marks in text stay alone.
    return re.sub(r"\?(?=\d)", "₹", value) if isinstance(value, str) else value


def load_companies(path: Path, inr_per_usd: float = 85.0) -> pd.DataFrame:
    df = _read_csv(path)
    # strip every string column (kills the trailing non-breaking spaces)
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].map(lambda v: v.replace("\xa0", " ").strip() if isinstance(v, str) else v)
    # put the rupee sign back in the money columns
    for col in ("valuation", "annual_revenue", "total_funding_till_date"):
        df[col] = df[col].map(_fix_rupee)

    # friendlier column names for the code
    df = df.rename(columns={"Company": "company", "Rank": "rank"})
    # rank is unique (1 to 118), so I use it as the company id everywhere
    df["id"] = df["rank"].astype(int)
    # map each specific sector to its group; anything not in my list lands in "Other"
    df["sector_group"] = df["primary_sector"].str.lower().map(_SECTOR_TO_GROUP).fillna("Other")
    # numeric versions of valuation and headcount so we can sort by them later
    df["valuation_usd_m"] = df["valuation"].map(lambda v: parse_money_usd_m(v, inr_per_usd))
    df["employees"] = df["employee_count"].map(parse_employees)
    df["unicorn_year"] = df["company_info"].map(parse_unicorn_year)
    # one block of text per company, this is what gets embedded for semantic search
    df["doc_text"] = df.apply(build_document, axis=1)
    return df.reset_index(drop=True)


def build_document(row: pd.Series) -> str:
    """The text that gets embedded for semantic search."""
    # a plain sentence about the company works better for embeddings than a pile of raw columns
    return (
        f"{row['company']} is a {row['sector_group']} company ({row['primary_sector']}) "
        f"based in {row['location']}, {row['country']}, founded in {row['founded_year']}, "
        f"currently at stage {row['stage']}. {row['company_background']}"
    )


def describe_company(row: pd.Series, with_description: bool = True) -> str:
    """Compact, LLM-friendly text block for one company (every field comes from the dataset)."""
    def val(x):
        # missing values become an explicit phrase, so the model says "not available" instead of guessing
        return "not available in dataset" if (x is None or (isinstance(x, float) and pd.isna(x)) or x == "") else x

    # two investor columns, skip whichever is empty
    investors = ", ".join(i for i in (row.get("top_investors_1"), row.get("top_investors_2"))
                          if isinstance(i, str) and i)
    # one labelled line per fact, easy for the LLM to read and quote from
    lines = [
        f"Company: {row['company']}",
        f"Sector: {row['sector_group']} / {row['primary_sector']}",
        f"HQ: {row['location']}, {row['country']}",
        f"Founded: {row['founded_year']} | Stage: {row['stage']} | Dataset rank: {row['rank']}",
        f"Valuation: {val(row['valuation'])} | Annual revenue: {val(row['annual_revenue'])}",
        f"Total funding: {val(row['total_funding_till_date'])} | Employees: {val(row['employee_count'])}",
        f"Latest round: {val(row['latest_funding_round'])} | Time to unicorn: {val(row['time_to_unicorn'])}",
        f"Top investors: {investors or 'not available in dataset'} | Round led by: {val(row['round_led_by'])}",
    ]
    # the long description is optional so we can keep prompts short when we don't need it
    if with_description:
        lines.append(f"Description: {row['company_background']}")
    return "\n".join(lines)