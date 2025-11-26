from json import load as json_load
from json import dump
from itertools import chain
from functools import reduce
from collections import Counter
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from glob import glob
from typing import Dict, List, Union
from entry_parser import EntryFactory, Cookie
from dataclasses import dataclass
"""
Each har file is parsed and analyzed into
a `WebsiteData` class instance whose fields
contain information crucial for the report.
"""

EntityDict = Dict[str, Dict[str, List[str]]]

CategoryDict = Dict[str, Dict[str, List[Dict[str, List[str]]]]]


@dataclass(frozen=True, slots=True, kw_only=True)
class WebsiteData:
    website: str
    num_requests: int
    num_3rd_party_requests: int
    distinct_web_domains: Dict[str, List[str]]
    distinct_entities: Dict[str, List[str]]
    cookies: List[Cookie]
    permission_policy: List[str]
    referrer_policy: List[Union[None, str]]
    accept_ch: List[Union[None, str]]
    redirections: list
    n_redirections: int

    @property
    def advertising_domains_present(self) -> bool:
        for v in self.distinct_web_domains.values():
            if "Advertising" in v:
                return True
        return False

    @property
    def analytics_domains_present(self) -> bool:
        for v in self.distinct_web_domains.values():
            if "Analytics" in v:
                return True
        return False


class WebsiteDataFactory:
    @staticmethod
    def _load_entities() -> EntityDict:
        f = open("entities.json", 'r')
        entities = json_load(f)
        f.close()
        return entities["entities"]

    @staticmethod
    def _entity_reverse_lookup(edict: EntityDict, domain: str) -> Union[None, str]:
        for k in edict.keys():
            if domain in edict[k]["properties"] or domain in edict[k]["resources"]:
                return k
        return None

    @staticmethod
    def _load_services() -> CategoryDict:
        f = open("services.json", 'r')
        services = json_load(f)
        f.close()
        return services["categories"]

    @staticmethod
    def _services_reverse_lookup(entity: str, cdict: CategoryDict):
        categories = []
        for category in cdict.keys():
            for d in cdict[category]:
                if entity in d.keys():
                    categories.append(category)
                    break
        return categories

    @staticmethod
    def _domains_reverse_lookup(domain: str, cdict: CategoryDict):
        categories = []
        for category in cdict.keys():
            for d in cdict[category]:
                for minidict in d.values():
                    for dilist in minidict.keys():
                        if str(domain) in dilist:
                            categories.append(category)
        return categories

    @staticmethod
    def measure_website_har(har_path: str) -> WebsiteData:
        '''
        Use this method to parse a har file and
        extract the WebsiteData object with the
        important information.
        '''
        edict = WebsiteDataFactory._load_entities()
        cdict = WebsiteDataFactory._load_services()
        entries = EntryFactory.entries_from_har(har_path)
        redirections = (list(filter(
            lambda e: e.response.status // 100 == 3, entries)))
        n_redirections = len(redirections)
        n_third_parties = len(
            list(filter(lambda x: x.is_3rd_party, entries)))
        home = entries[0].home_site
        web_domains = set(
            (e.request.web_domain for e in entries if e.request.web_domain != home)
        )
        entities = (WebsiteDataFactory._entity_reverse_lookup(edict, domain)
                    for domain in web_domains)
        entities = list(set(e for e in entities if e is not None))
        entities = {
            e: WebsiteDataFactory._services_reverse_lookup(e, cdict) for e in entities

        }
        domains = {
            d: WebsiteDataFactory._domains_reverse_lookup(d, cdict) for d in web_domains
        }
        cookies = [e.request.cookies + e.response.set_cookies for e in entries]
        referrer_policy = set(list(
            map(lambda x: x.response.referrer_policy, entries)))
        accept_ch = set(list(
            map(lambda x: x.response.accept_ch, entries)))
        for entry in entries:
            break
        permissions = [x.response.permission_policy for x in entries]
        metrics = WebsiteData(

            website=home,
            num_requests=len(entries),
            num_3rd_party_requests=n_third_parties,
            distinct_web_domains=domains,
            distinct_entities=entities,
            cookies=(list(chain.from_iterable(cookies))),
            permission_policy=permissions,
            referrer_policy=list(referrer_policy),
            accept_ch=list(accept_ch),
            redirections=redirections,
            n_redirections=n_redirections)
        return metrics


if __name__ == "__main__":
    def extract(x, datalist): return [getattr(datum, x) for datum in datalist]
    data_folder = "data"
    data = {
        "Crawl-Accept": None,
        "Crawl-Reject": None,
        "Crawl-Block": None
    }
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    accepts = []
    rejects = []
    blocks = []
    for fp in tqdm(glob("../**/*.har", recursive=True), desc="Analyzing HAR files"):
        metric = WebsiteDataFactory.measure_website_har(fp)
        if "accept" in fp:
            accepts.append(metric)
        elif "reject" in fp:
            rejects.append(metric)
        elif "block" in fp:
            blocks.append(metric)
        else:
            raise ValueError(f"Unexpected HAR file {fp}")
    # exercise 1
    plt.figure()
    nreq_a = [site.num_requests for site in accepts]
    nreq_r = [site.num_requests for site in rejects]
    nreq_b = [site.num_requests for site in blocks]
    data["Crawl-Accept"] = nreq_a
    data["Crawl-Reject"] = nreq_r
    data["Crawl-Block"] = nreq_b
    nreq_df = pd.DataFrame(data)
    sns.boxplot(data=data)
    plt.title("Number of requests (1a)")
    plt.savefig(f"{data_folder}/1a.png")

    nreq_a = [site.num_3rd_party_requests for site in accepts]
    nreq_r = [site.num_3rd_party_requests for site in rejects]
    nreq_b = [site.num_3rd_party_requests for site in blocks]
    data["Crawl-Accept"] = nreq_a
    data["Crawl-Reject"] = nreq_r
    data["Crawl-Block"] = nreq_b
    nreq_df = pd.DataFrame(data)
    sns.boxplot(data=data)
    plt.title("Number of 3rd party requests (1b)")
    plt.savefig(f"{data_folder}/1b.png")

    nreq_a = [len(site.distinct_web_domains) for site in accepts]
    nreq_r = [len(site.distinct_web_domains) for site in rejects]
    nreq_b = [len(site.distinct_web_domains) for site in blocks]
    data["Crawl-Accept"] = nreq_a
    data["Crawl-Reject"] = nreq_r
    data["Crawl-Block"] = nreq_b
    nreq_df = pd.DataFrame(data)
    sns.boxplot(data=data)
    plt.title("Number of distinct 3rd party domains (1c)")
    plt.savefig(f"{data_folder}/1c.png")

    nreq_a = [len(site.distinct_entities) for site in accepts]
    nreq_r = [len(site.distinct_entities) for site in rejects]
    nreq_b = [len(site.distinct_entities) for site in blocks]
    data["Crawl-Accept"] = nreq_a
    data["Crawl-Reject"] = nreq_r
    data["Crawl-Block"] = nreq_b
    nreq_df = pd.DataFrame(data)
    sns.boxplot(data=data)
    plt.title("Number of distinct 3rd party domains (1d)")
    plt.savefig(f"{data_folder}/1d.png")

    # exercise 2

    accept_stats = {
        "Min": {
            "Total Requests": np.min(extract("num_requests", accepts)),
            "3rd Party Requests": np.min(extract("num_3rd_party_requests", accepts)),
            "Distinct Web Domains": np.min([len(d) for d in extract("distinct_web_domains", accepts)]),
            "Distinct Entities": np.min([len(d) for d in extract("distinct_entities", accepts)]),
        },
        "Max": {
            "Total Requests": np.max(extract("num_requests", accepts)),
            "3rd Party Requests": np.max(extract("num_3rd_party_requests", accepts)),
            "Distinct Web Domains": np.max([len(d) for d in extract("distinct_web_domains", accepts)]),
            "Distinct Entities": np.max([len(d) for d in extract("distinct_entities", accepts)]),
        },
        "Median": {
            "Total Requests": np.median(extract("num_requests", accepts)),
            "3rd Party Requests": np.median(extract("num_3rd_party_requests", accepts)),
            "Distinct Web Domains": np.median([len(d) for d in extract("distinct_web_domains", accepts)]),
            "Distinct Entities": np.median([len(d) for d in extract("distinct_entities", accepts)]),
        },
    }

    reject_stats = {
        "Min": {
            "Total Requests": np.min(extract("num_requests", rejects)),
            "3rd Party Requests": np.min(extract("num_3rd_party_requests", rejects)),
            "Distinct Web Domains": np.min([len(d) for d in extract("distinct_web_domains", rejects)]),
            "Distinct Entities": np.min([len(d) for d in extract("distinct_entities", rejects)]),
        },
        "Max": {
            "Total Requests": np.max(extract("num_requests", rejects)),
            "3rd Party Requests": np.max(extract("num_3rd_party_requests", rejects)),
            "Distinct Web Domains": np.max([len(d) for d in extract("distinct_web_domains", rejects)]),
            "Distinct Entities": np.max([len(d) for d in extract("distinct_entities", rejects)]),
        },
        "Median": {
            "Total Requests": np.median(extract("num_requests", rejects)),
            "3rd Party Requests": np.median(extract("num_3rd_party_requests", rejects)),
            "Distinct Web Domains": np.median([len(d) for d in extract("distinct_web_domains", rejects)]),
            "Distinct Entities": np.median([len(d) for d in extract("distinct_entities", rejects)]),
        },
    }
    block_stats = {
        "Min": {
            "Total Requests": np.min(extract("num_requests", blocks)),
            "3rd Party Requests": np.min(extract("num_3rd_party_requests", blocks)),
            "Distinct Web Domains": np.min([len(d) for d in extract("distinct_web_domains", blocks)]),
            "Distinct Entities": np.min([len(d) for d in extract("distinct_entities", blocks)]),
        },
        "Max": {
            "Total Requests": np.max(extract("num_requests", blocks)),
            "3rd Party Requests": np.max(extract("num_3rd_party_requests", blocks)),
            "Distinct Web Domains": np.max([len(d) for d in extract("distinct_web_domains", blocks)]),
            "Distinct Entities": np.max([len(d) for d in extract("distinct_entities", blocks)]),
        },
        "Median": {
            "Total Requests": np.median(extract("num_requests", blocks)),
            "3rd Party Requests": np.median(extract("num_3rd_party_requests", blocks)),
            "Distinct Web Domains": np.median([len(d) for d in extract("distinct_web_domains", blocks)]),
            "Distinct Entities": np.median([len(d) for d in extract("distinct_entities", blocks)]),
        },
    }
    statistics = {
        "Accept": accept_stats,
        "Reject": reject_stats,
        "Block": block_stats,
    }

    pd.DataFrame(statistics).to_json(f"{data_folder}/e2.json")
    # exercise 3
    ads = {
        "Accept": sum(extract("advertising_domains_present", accepts)),
        "Reject": sum(extract("advertising_domains_present", rejects)),
        "Block": sum(extract("advertising_domains_present", blocks)),
    }

    analytics = {
        "Accept": sum(extract("analytics_domains_present", accepts)),
        "Reject": sum(extract("analytics_domains_present", rejects)),
        "Block": sum(extract("analytics_domains_present", blocks)),
    }

    data = {
        "Advertising": ads,
        "Analytics": analytics,
    }
    data = pd.DataFrame(data)

    plt.figure(num=randint(0, 2**32))
    sns.barplot(data=data.Advertising)
    plt.savefig(f"{data_folder}/3a.png")

    plt.figure(num=randint(0, 2**32))
    sns.barplot(data=data.Analytics)
    plt.savefig(f"{data_folder}/3b.png")

    # exercise 4

    us = [
        "msn.com",
        "nytimes.com",
        "cnn.com",
        "nbcnews.com",
        "foxnews.com",
        "nypost.com",
        "apnews.com",
        "cnbc.com",
        "newsweek.com",
        "cbsnews.com",
        "wsj.com",
        "washingtonpost.com",
        "reuters.com",
        "businessinsider.com",
        "huffpost.com",
        "buzzfeed.com",
        "axios.com",
        "bloomberg.com",
        "latimes.com",
        "msnbc.com",
    ]

    accepts_us = [w for w in accepts if w.website in us]
    accepts_eu = [w for w in accepts if w.website not in us]

    accept_stats_eu = {
        "Min": {
            "Total Requests": np.min(extract("num_requests", accepts_eu)),
            "3rd Party Requests": np.min(extract("num_3rd_party_requests", accepts_eu)),
            "Distinct Web Domains": np.min([len(d) for d in extract("distinct_web_domains", accepts_eu)]),
            "Distinct Entities": np.min([len(d) for d in extract("distinct_entities", accepts_eu)]),
        },
        "Max": {
            "Total Requests": np.max(extract("num_requests", accepts_eu)),
            "3rd Party Requests": np.max(extract("num_3rd_party_requests", accepts_eu)),
            "Distinct Web Domains": np.max([len(d) for d in extract("distinct_web_domains", accepts_eu)]),
            "Distinct Entities": np.max([len(d) for d in extract("distinct_entities", accepts_eu)]),
        },
        "Median": {
            "Total Requests": np.median(extract("num_requests", accepts_eu)),
            "3rd Party Requests": np.median(extract("num_3rd_party_requests", accepts_eu)),
            "Distinct Web Domains": np.median([len(d) for d in extract("distinct_web_domains", accepts_eu)]),
            "Distinct Entities": np.median([len(d) for d in extract("distinct_entities", accepts_eu)]),
        },
    }

    accept_stats_us = {
        "Min": {
            "Total Requests": np.min(extract("num_requests", accepts_us)),
            "3rd Party Requests": np.min(extract("num_3rd_party_requests", accepts_us)),
            "Distinct Web Domains": np.min([len(d) for d in extract("distinct_web_domains", accepts_us)]),
            "Distinct Entities": np.min([len(d) for d in extract("distinct_entities", accepts_us)]),
        },
        "Max": {
            "Total Requests": np.max(extract("num_requests", accepts_us)),
            "3rd Party Requests": np.max(extract("num_3rd_party_requests", accepts_us)),
            "Distinct Web Domains": np.max([len(d) for d in extract("distinct_web_domains", accepts_us)]),
            "Distinct Entities": np.max([len(d) for d in extract("distinct_entities", accepts_us)]),
        },
        "Median": {
            "Total Requests": np.median(extract("num_requests", accepts_us)),
            "3rd Party Requests": np.median(extract("num_3rd_party_requests", accepts_us)),
            "Distinct Web Domains": np.median([len(d) for d in extract("distinct_web_domains", accepts_us)]),
            "Distinct Entities": np.median([len(d) for d in extract("distinct_entities", accepts_us)]),
        },
    }

    accepts_us_eu = {
        "Accept-US": accept_stats_us,
        "Accept-EU": accept_stats_eu,
    }
    pd.DataFrame(accepts_us_eu).to_json(f"{data_folder}/e4.json")

    # exercise 5

    ads_eu_us = {
        "US": sum(extract("advertising_domains_present", accepts_us)),
        "EU": sum(extract("advertising_domains_present", accepts_eu)),
    }

    analytics_eu_us = {
        "US": sum(extract("analytics_domains_present", accepts_us)),
        "EU": sum(extract("analytics_domains_present", accepts_eu)),
    }

    data = {
        "Advertising": ads_eu_us,
        "Analytics": analytics_eu_us,
    }
    data = pd.DataFrame(data)

    plt.figure(num=randint(0, 2**32))
    sns.barplot(data=data.Advertising)
    plt.savefig(f"{data_folder}/5a.png")

    plt.figure(num=randint(0, 2**32))
    sns.barplot(data=data.Analytics)
    plt.savefig(f"{data_folder}/5b.png")


# exercise 6

    accept_cookies = []
    for fp in tqdm(glob("test_hars/**/*.json", recursive=True), desc="Analyzing browser cookies"):
        f = open(fp, 'r')
        cookies = json_load(f)
        f.close()
        names = [c.strip().split("=")[0]
                 for c in cookies["document_cookie"].split(";")]
        if "accept" in fp:
            accept_cookies += names

        cookies = set(cookies)
        for cookie in cookies:
            for website in accepts:
                for c in website.cookies:
                    if c.name == cookie:
                        print(c.domain)


# exercise 7

    domain_dicts = extract("distinct_web_domains", accepts)
    domains = Counter(reduce(lambda x, acc: x + acc,
                             list(map(lambda x: list(x.keys()), domain_dicts))))

    most_common_10 = domains.most_common(10)
    names = [c[0] for c in most_common_10]
    values = [c[1] for c in most_common_10]
    collapsed = reduce(lambda x, acc: x | acc, domain_dicts)
    categories = [collapsed[name] for name in names]

    most_common_cat = {
        "Third-party domain":  names,
        "Number of distinct websites": values,
        "Categories": categories
    }

    pd.DataFrame(most_common_cat).to_json(f"{data_folder}/7.json")

# exercise 8
    nl = [
        "nos.nl",
        "nu.nl",
        "ad.nl",
        "telegraaf.nl",
        "nltimes.nl",
    ]
    de = ["rtl.de", "n-tv.de", "dw.com"]
    fr = ["lefigaro.fr",
          "lemonde.fr",
          "france24.com"]
    it = [
        "libero.it",
        "repubblica.it",
        "corriere.it",
        "gazzetta.it",]
    uk = ["theguardian.com",
          "dailymail.co.uk",
          "the-sun.com",
          "bbc.co.uk",
          "skynews.com"]

    def country_code(site):
        if site.website in nl:
            return "nl"
        elif site.website in de:
            return "de"
        elif site.website in fr:
            return "fr"
        elif site.website in it:
            return "it"
        elif site.website in uk:
            return "uk"
        return 'us'
    accepts.sort(key=lambda x: len(
        x.distinct_web_domains.keys()), reverse=True)
    rejects.sort(key=lambda x: len(
        x.distinct_web_domains.keys()), reverse=True)
    blocks.sort(key=lambda x: len(x.distinct_web_domains.keys()), reverse=True)

    top_accepts = accepts[:10]
    names_accepts = [x.website for x in top_accepts]
    codes_accepts = list(map(country_code, top_accepts))
    number_accepts = [len(list(website.distinct_web_domains.keys()))
                      for website in top_accepts]

    top_rejects = rejects[:10]
    names_rejects = [x.website for x in top_rejects]
    codes_rejects = list(map(country_code, top_rejects))
    number_rejects = [len(list(website.distinct_web_domains.keys()))
                      for website in top_rejects]
    top_blocks = blocks[:10]
    names_blocks = [x.website for x in top_blocks]
    codes_blocks = list(map(country_code, top_blocks))
    number_blocks = [len(list(website.distinct_web_domains.keys()))
                     for website in top_blocks]

    data = {
        "Accept": {
            "Name": names_accepts,
            "Country code": codes_accepts,
            "Number of distinct third-party domains": number_accepts
        },
        "Reject": {
            "Name": names_rejects,
            "Country code": codes_rejects,
            "Number of distinct third-party domains": number_rejects
        },
        "Blocks": {
            "Name": names_blocks,
            "Country code": codes_blocks,
            "Number of distinct third-party domains": number_blocks
        },
    }
    data = pd.DataFrame(data)
    data.to_json(f"{data_folder}/e8.json")

    # exercise 10
    permissions_accepts = [d.permission_policy for d in accepts]
    permissions_rejects = [d.permission_policy for d in rejects]
    permissions_blocks = [d.permission_policy for d in blocks]

    pd.DataFrame({
        "Accept": permissions_accepts,
        "Reject": permissions_rejects,
        "Block": permissions_blocks
    }).to_json(f"{data_folder}/e10.json")

    # exercise 11

    referrer_accepts = Counter(list(filter(lambda x: x is not None, [l[0] for l in [
        d.referrer_policy for d in accepts]])))

    referrer_rejects = Counter(list(filter(lambda x: x is not None, [l[0] for l in [
        d.referrer_policy for d in rejects]])))
    referrer_blocks = Counter(list(filter(lambda x: x is not None, [l[0] for l in [
        d.referrer_policy for d in blocks]])))
    data = {
        "Accept": referrer_accepts,
        "Reject": referrer_rejects,
        "Block": referrer_blocks
    }
    pd.DataFrame(data).to_json(f"{data_folder}/e11.json")

    # exercise 12

    ch_accepts = Counter(list(filter(lambda x: x is not None, [l[0] for l in [
        d.accept_ch for d in accepts]])))

    ch_rejects = Counter(list(filter(lambda x: x is not None, [l[0] for l in [
        d.accept_ch for d in rejects]])))
    ch_blocks = Counter(list(filter(lambda x: x is not None, [l[0] for l in [
        d.accept_ch for d in blocks]])))

    data = {
        "Accept": ch_accepts,
        "Reject": ch_rejects,
        "Block":  ch_blocks
    }
    pd.DataFrame(data).to_json(f"{data_folder}/e12.json")

    # exercise 13

    lookup = WebsiteDataFactory._entity_reverse_lookup
    entities = WebsiteDataFactory._load_entities()

    redirection_accepts = reduce(
        lambda acc, x: x+acc, [(w.redirections) for w in accepts])
    redirection_accepts = list(
        filter(lambda x: x.home_site != x.request.web_domain, redirection_accepts))
    redirection_accepts = list(
        map(
            lambda red: (
                red.home_site, red.request.web_domain), redirection_accepts
        )
    )
    redirection_accepts = list(
        filter(
            lambda x: lookup(entities, x[0]) is not None, redirection_accepts
        ))

    redirection_accepts = list(
        map(
            lambda x: [x[0], lookup(entities, x[0]),
                       x[1], lookup(entities, x[1])], redirection_accepts
        )
    )
    redirection_rejects = reduce(
        lambda acc, x: x+acc, [(w.redirections) for w in rejects])
    redirection_rejects = list(
        filter(lambda x: x.home_site != x.request.web_domain, redirection_rejects))
    redirection_rejects = list(
        map(
            lambda red: (
                red.home_site, red.request.web_domain), redirection_rejects
        )
    )
    redirection_rejects = list(
        filter(
            lambda x: lookup(entities, x[0]) is not None, redirection_rejects
        ))

    redirection_rejects = list(
        map(
            lambda x: [x[0], lookup(entities, x[0]),
                       x[1], lookup(entities, x[1])], redirection_rejects
        )
    )
    redirection_blocks = reduce(
        lambda acc, x: x+acc, [(w.redirections) for w in blocks])
    redirection_blocks = list(
        filter(lambda x: x.home_site != x.request.web_domain, redirection_blocks))
    redirection_blocks = list(
        map(
            lambda red: (
                red.home_site, red.request.web_domain), redirection_blocks
        )
    )
    redirection_blocks = list(
        filter(
            lambda x: lookup(entities, x[0]) is not None, redirection_blocks
        ))

    redirection_blocks = list(
        map(
            lambda x: [x[0], lookup(entities, x[0]),
                       x[1], lookup(entities, x[1])], redirection_blocks
        )
    )

    data = {
        "Accept": redirection_accepts,
        "Reject": redirection_rejects,
        "Block": redirection_blocks,
    }

    with open(f"{data_folder}/e13.json", 'w') as f:
        dump(data, f)
