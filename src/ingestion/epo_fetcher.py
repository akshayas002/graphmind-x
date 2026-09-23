"""
Week 1, Step: pull a first batch of real Indian-patent bibliographic records
from the EPO's Open Patent Services (OPS) API.

Why EPO OPS and not IP India directly: IP India (the Indian Patent Office)
has no public API. Their only public system, InPASS, is a JS-heavy,
CAPTCHA-protected web portal meant for humans, not programmatic access.
EPO OPS is official, free (4 GB/month), and includes Indian patent
bibliographic + legal-status data via INPADOC (the EPO's worldwide patent
family database, which IP India reports into).

Trade-off to know going in: OPS gives strong bibliographic data (title,
applicants, inventors, IPC classification, publication/application numbers,
family links) but generally thinner full specification text for India-origin
filings than, say, USPTO gives for US patents. Fine for the graph's
structure; the Explorer/Verifier agents (later weeks) do the deeper text work.

Usage:
    python -m src.ingestion.epo_fetcher
    python -m src.ingestion.epo_fetcher --probe "battery" "lithium ion"
    python -m src.ingestion.epo_fetcher --loose
"""
import argparse
import base64
import json
import sys
import time
from datetime import datetime, timezone

import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from src import config


class EPOFetcher:
    def __init__(
        self,
        consumer_key: str = config.EPO_OPS_CONSUMER_KEY,
        consumer_secret: str = config.EPO_OPS_CONSUMER_SECRET,
    ):
        if not consumer_key or not consumer_secret:
            raise ValueError(
                "EPO_OPS_CONSUMER_KEY / EPO_OPS_CONSUMER_SECRET are not set. "
                "Copy .env.example to .env, register a free app at "
                "https://developers.epo.org, and add your credentials there."
            )
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self._access_token = None
        self._token_expiry = 0
        self.session = requests.Session()

    def _authenticate(self):
        """OAuth2 client-credentials flow. Tokens are short-lived (~20 min)."""
        credentials = f"{self.consumer_key}:{self.consumer_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        resp = self.session.post(
            config.EPO_OPS_AUTH_URL,
            headers={
                "Authorization": f"Basic {encoded}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"grant_type": "client_credentials"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        self._access_token = data["access_token"]
        self._token_expiry = time.time() + int(data.get("expires_in", 1200)) - 60

    def _ensure_token(self):
        if not self._access_token or time.time() >= self._token_expiry:
            self._authenticate()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20))
    def _get_page(self, cql_query: str, range_start: int, range_end: int) -> dict:
        self._ensure_token()
        resp = self.session.get(
            config.EPO_OPS_SEARCH_BIBLIO_URL,
            params={"q": cql_query},
            headers={
                "Authorization": f"Bearer {self._access_token}",
                "Accept": "application/json",
                "X-OPS-Range": f"{range_start}-{range_end}",
                "Range": f"{range_start}-{range_end}",
            },
            timeout=30,
        )
        if resp.status_code in (401, 403):
            self._authenticate()
            resp = self.session.get(
                config.EPO_OPS_SEARCH_BIBLIO_URL,
                params={"q": cql_query},
                headers={
                    "Authorization": f"Bearer {self._access_token}",
                    "Accept": "application/json",
                    "Range": f"{range_start}-{range_end}",
                },
                timeout=30,
            )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def build_cql_query(keyword: str, country_code: str, loose: bool = False) -> str:
        """
        loose=False (default): exact-phrase match — ta="solid state battery"
        loose=True: all words must appear (any order/position), much broader —
        ta=solid AND ta=state AND ta=battery
        """
        if loose:
            terms = keyword.split()
            phrase_part = " AND ".join(f"ta={t}" for t in terms)
            return f"pn={country_code} AND {phrase_part}"
        return f'pn={country_code} AND ta="{keyword}"'

    def probe(self, keywords: list[str], country_code: str, loose: bool = False) -> dict[str, int | None]:
        """
        Quick reconnaissance: check total-result-count for several candidate
        keywords without downloading any actual records (range 1-1 = cheapest
        possible request). Use this before committing to a full pull.
        """
        counts: dict[str, int | None] = {}
        for kw in keywords:
            cql_query = self.build_cql_query(kw, country_code, loose=loose)
            try:
                data = self._get_page(cql_query, 1, 1)
                counts[kw] = self._extract_total_count(data)
            except requests.HTTPError as e:
                counts[kw] = None
                print(f"  '{kw}': HTTP error — {e}")
        return counts

    def search(self, keyword: str, country_code: str, limit: int = 200, loose: bool = False) -> list[dict]:
        """
        Search Indian-scoped patents by keyword.

        NOTE ON COUNTRY FILTERING: this restricts via `pn={country_code}`
        (publication-number country prefix), which is the commonly documented
        pattern for country-of-publication filtering in EPO's CQL. If your
        first small run comes back with non-Indian results, sanity-check the
        query on Espacenet's Advanced Search UI (worldwide.espacenet.com)
        before trusting a large pull — OPS's CQL behavior around country
        prefixes has shifted before.
        """
        cql_query = self.build_cql_query(keyword, country_code, loose=loose)
        page_size = 25  # conservative default page size for the free tier
        pages: list[dict] = []
        fetched = 0
        real_total: int | None = None

        pbar = tqdm(total=limit, desc=f"Fetching EPO OPS '{keyword}' ({country_code})")
        try:
            while fetched < limit:
                start = fetched + 1
                end = min(fetched + page_size, limit)
                data = self._get_page(cql_query, start, end)
                pages.append(data)

                if real_total is None:
                    real_total = self._extract_total_count(data)
                    if real_total is not None:
                        pbar.total = min(limit, real_total)
                        pbar.refresh()
                        if real_total < limit:
                            tqdm.write(
                                f"Note: EPO OPS reports only {real_total} total result(s) for this "
                                f"query — fewer than your target of {limit}. Not an error; this "
                                f"query is just that narrow. Consider broadening the keyword."
                            )

                docs_this_page = self._count_documents(data)
                fetched_before = fetched
                fetched += docs_this_page
                pbar.update(fetched - fetched_before)

                if docs_this_page == 0:
                    break
                if real_total is not None and fetched >= real_total:
                    break
        finally:
            pbar.close()

        return pages

    @staticmethod
    def _count_documents(page: dict) -> int:
        """Count actual exchange-document entries in a page (handles the OPS
        single-item-dict vs multi-item-list quirk — see entity_extractor.as_list)."""
        try:
            docs = page["ops:world-patent-data"]["ops:biblio-search"]["ops:search-result"][
                "exchange-documents"
            ]["exchange-document"]
        except (KeyError, TypeError):
            return 0
        return len(docs) if isinstance(docs, list) else 1

    @staticmethod
    def _extract_total_count(page: dict) -> int | None:
        try:
            biblio_search = page["ops:world-patent-data"]["ops:biblio-search"]
            return int(biblio_search.get("@total-result-count", 0))
        except (KeyError, TypeError, ValueError):
            return None

    def save_raw(self, pages: list[dict], keyword: str, country_code: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_keyword = keyword.replace(" ", "_")
        out_path = config.DATA_RAW_DIR / f"epo_{country_code}_{safe_keyword}_{timestamp}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pages, f, indent=2, ensure_ascii=False)
        return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Fetch Indian patent data from EPO OPS.")
    parser.add_argument(
        "--probe",
        nargs="+",
        metavar="KEYWORD",
        help="Check total-result-count for one or more keywords without downloading "
             "any records — fast reconnaissance before a full pull. "
             'Example: --probe "battery" "lithium ion" "energy storage"',
    )
    parser.add_argument(
        "--loose",
        action="store_true",
        help="Match all words in the keyword in any order/position, instead of requiring "
             "the exact phrase. Broadens results significantly. Applies to --probe too.",
    )
    args = parser.parse_args()

    fetcher = EPOFetcher()
    country_code = config.EPO_COUNTRY_CODE

    if args.probe:
        print(f"Probing EPO OPS for {len(args.probe)} keyword(s), country='{country_code}', "
              f"loose={args.loose}...\n")
        counts = fetcher.probe(args.probe, country_code, loose=args.loose)
        print("\nResults:")
        for kw, count in counts.items():
            display = count if count is not None else "error"
            print(f"  '{kw}': {display} total result(s)")
        print("\nPick a keyword with enough volume, then run without --probe to fetch it "
              "(set PATENT_SEARCH_KEYWORD in .env, or see the loose-query tip below).")
        return

    keyword = config.PATENT_SEARCH_KEYWORD
    limit = config.INGESTION_SAMPLE_SIZE

    print(f"Searching EPO OPS for '{keyword}' restricted to country '{country_code}' "
          f"(target: {limit} records, loose={args.loose})...")
    print("Tip: on your first run, try limit=10 in .env and eyeball the raw JSON before "
          "pulling the full batch — confirms the country filter is doing what you expect.")

    try:
        pages = fetcher.search(keyword=keyword, country_code=country_code, limit=limit, loose=args.loose)
    except requests.HTTPError as e:
        print(f"\nHTTP error from EPO OPS: {e}")
        print("Common causes: invalid consumer key/secret, malformed CQL query, "
              "or monthly data quota exhausted.")
        sys.exit(1)

    if not pages:
        print("No pages returned — check your query and credentials.")
        sys.exit(1)

    out_path = fetcher.save_raw(pages, keyword, country_code)
    total_docs = sum(fetcher._count_documents(p) for p in pages)
    print(f"\nSaved {len(pages)} page(s), {total_docs} actual patent record(s), to {out_path}")
    if total_docs < 20:
        print(
            "That's a small result set. Try: python -m src.ingestion.epo_fetcher --probe "
            '"battery" "lithium ion" "energy storage" "electric vehicle" to find a broader '
            "keyword before committing to a full pull. Add --loose to relax exact-phrase matching."
        )
    print(f"Next: python -m src.extraction.entity_extractor {out_path}")


if __name__ == "__main__":
    main()