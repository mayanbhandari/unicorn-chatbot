#Does the exact, non-AI work on the data: filtering and sorting companies with pandas, and fuzzy company-name lookup. We need it so lists and rankings are always correct; the LLM only picks the filters, this file actually runs them.

from __future__ import annotations

import re
from typing import Iterable, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field
from rapidfuzz import fuzz, process

from .data_loader import normalise_country, normalise_locations

# The only things a user can sort by, and which dataframe column each one means.
SortKey = Literal["valuation", "employees", "founded_year", "rank"]
_SORT_COLUMNS = {"valuation": "valuation_usd_m", "employees": "employees",
                 "founded_year": "founded_year", "rank": "rank"}


class Filters(BaseModel):
    """Structured filters extracted from the user's question."""
    # The router (LLM) fills this in from the question. Every field is optional, empty means "don't filter on this".
    # The descriptions below are read by the LLM, so they double as instructions for it.
    sector_groups: list[str] = Field(default_factory=list, description="Values from the allowed sector_group list")
    primary_sector_keywords: list[str] = Field(default_factory=list, description="Substrings of primary_sector, e.g. 'payments'")
    locations: list[str] = Field(default_factory=list, description="HQ cities, user wording is fine (Bangalore, NCR...)")
    countries: list[str] = Field(default_factory=list)
    stages: list[str] = Field(default_factory=list, description="e.g. 'Series E', 'Public', 'Acquired'")
    founded_after: Optional[int] = None
    founded_before: Optional[int] = None
    sort_by: Optional[SortKey] = None
    sort_desc: bool = True        # biggest first by default
    limit: Optional[int] = None   # "top 5", "the most valuable" (limit 1)

    def is_empty(self) -> bool:
        # True when nothing is being filtered (sorting and limit don't count as filters)
        return not any([self.sector_groups, self.primary_sector_keywords, self.locations, self.countries,
                        self.stages, self.founded_after, self.founded_before])


def _ci_in(series: pd.Series, wanted: Iterable[str]) -> pd.Series:
    # "case-insensitive is in": lowercase both sides so "fintech" matches "Fintech"
    wanted_l = {w.strip().lower() for w in wanted}
    return series.str.lower().isin(wanted_l)


def apply_filters(df: pd.DataFrame, f: Filters, candidate_ids: Optional[list[int]] = None) -> pd.DataFrame:
    """Apply filters (AND across fields, OR within a field). `candidate_ids` restricts to a previous result set."""
    out = df
    # follow-up questions ("which of these...") pass in the ids from the last answer, so we only search inside those
    if candidate_ids is not None:
        out = out[out["id"].isin(candidate_ids)]
    if f.sector_groups:
        out = out[_ci_in(out["sector_group"], f.sector_groups)]
    if f.primary_sector_keywords:
        # join keywords with | so any one can match, and escape regex symbols to treat them literally
        pattern = "|".join(re.escape(k.strip().lower()) for k in f.primary_sector_keywords if k.strip())
        if pattern:
            out = out[out["primary_sector"].str.lower().str.contains(pattern, regex=True, na=False)]
    if f.locations:
        # translate "Bangalore" into "Bengaluru" etc. before comparing
        out = out[_ci_in(out["location"], normalise_locations(f.locations))]
    if f.countries:
        out = out[_ci_in(out["country"], [normalise_country(c) for c in f.countries])]
    if f.stages:
        out = out[_ci_in(out["stage"], f.stages)]
    if f.founded_after is not None:
        out = out[out["founded_year"] > f.founded_after]
    if f.founded_before is not None:
        out = out[out["founded_year"] < f.founded_before]
    if f.sort_by:
        col = _SORT_COLUMNS[f.sort_by]
        # companies with no value for that column (like a missing valuation) go to the bottom
        out = out.sort_values(col, ascending=not f.sort_desc, na_position="last")
    else:
        out = out.sort_values("rank", ascending=True)     # dataset order: rank 1 first
    # a limit of 1 is how "the most valuable company" gets answered
    if f.limit and f.limit > 0:
        out = out.head(f.limit)
    return out


def find_companies(df: pd.DataFrame, names: list[str], cutoff: int = 82) -> tuple[pd.DataFrame, list[str]]:
    """Fuzzy-match company names. Returns (matched rows, names that could not be matched)."""
    choices = df["company"].tolist()
    matched_ids: list[int] = []
    missing: list[str] = []
    for name in names:
        # fuzzy match so small typos still work ("razor pay" finds Razorpay); below the cutoff counts as not found
        hit = process.extractOne(name, choices, scorer=fuzz.WRatio, score_cutoff=cutoff)
        if hit:
            matched_ids.append(int(df.loc[df["company"] == hit[0], "id"].iloc[0]))
        else:
            missing.append(name)
    rows = df[df["id"].isin(matched_ids)]
    # keep the order the user mentioned them in
    rows = rows.set_index("id").loc[list(dict.fromkeys(matched_ids))].reset_index() if matched_ids else rows
    return rows, missing


def closest_names(df: pd.DataFrame, name: str, n: int = 3) -> list[str]:
    # used for the "Did you mean...?" hint when a company isn't found; returns the n nearest names
    return [c for c, _score, _i in process.extract(name, df["company"].tolist(), scorer=fuzz.WRatio, limit=n)]
