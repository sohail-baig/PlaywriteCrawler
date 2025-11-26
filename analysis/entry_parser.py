from enum import Enum
from pathlib import Path
from urllib.parse import urlparse
from dataclasses import dataclass
from json import load as json_load
from typing import Union, List, Dict, Iterator
from publicsuffix2 import get_sld

'''
Do not use the dataclasses directly.
Use the method `entries_from_har` described
in the `EntryFactory` class to
parse har files into the appropriate
format.
'''
Method: Enum = Enum('Method', [("GET", 0), ("POST", 1),
                               ("HEAD", 2), ("PUT", 3),
                               ("CONNECT", 4), ("OPTIONS", 5),
                               ("DELETE", 6), ("TRACE", 7),
                               ("PATCH", 9)])


EntryDir = Dict[str, Union[str, List[str]]]


@dataclass(frozen=True, slots=True, kw_only=True)
class Cookie:
    name: str
    value: str
    domain: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Request:
    method: Method
    url: str
    cookies:  List[Cookie]

    @property
    def web_domain(self):
        return get_sld(urlparse(self.url).netloc)


@dataclass(frozen=True, slots=True, kw_only=True)
class Response:
    status: int
    referrer_policy: Union[None, str]
    permission_policy: Union[None, str]
    accept_ch: Union[None, str]
    set_cookies: List[Cookie]
    redirect_url: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Entry:
    home_site: str
    request: Request
    response: Response

    @property
    def is_3rd_party(self):
        return self.home_site == get_sld(urlparse(self.request.url).netloc)


class EntryFactory:
    @staticmethod
    def _parse_request(home_site: str, edir: EntryDir) -> Request:
        request = edir["request"]
        method = getattr(Method, request["method"].upper())
        url = request['url']
        headers = request["headers"]
        cookies = []
        name = None,
        value = None

        for header in headers:
            if header["name"] == "cookie":

                cookies_list = header["value"].split(";")
                for c in cookies_list:
                    try:
                        name, value = c.split("=", 1)
                        cookies.append(
                            Cookie(name=name.strip(), value=value.strip(), domain=get_sld(urlparse(url).netloc)))
                    except ValueError:
                        cookies.append(
                            Cookie(name=c, value=None, domain=get_sld(urlparse(url).netloc)))

            elif header["name"] == "Referrer":
                print(header)

        return Request(method=method, url=url, cookies=cookies)

    @ staticmethod
    def _parse_response(rdir: EntryDir) -> Response:
        rsp = rdir["response"]
        url = rdir["request"]["url"]
        redirect = rsp["redirectURL"]
        status = rsp["status"]
        headers = rsp["headers"]
        cookie_objects = []
        ref_policy = None
        perm_policy = None
        accept_ch = None
        for header in headers:
            if header["name"] == "set-cookie":

                try:
                    cookie_name, cookie_value = header["value"].split(";")[
                        0].split("=", 1)
                    cookie_objects.append(
                        Cookie(name=cookie_name.strip(),
                               value=cookie_value.strip(), domain=get_sld(urlparse(url).netloc)))

                except ValueError:
                    cookie_objects.append(
                        Cookie(name=header["value"], value=header["value"], domain=get_sld(urlparse(url).netloc)))

            elif header["name"] == "Referrer-Policy":
                ref_policy = header["value"]
            elif header["name"] == "Accept-CH":
                accept_ch = header["value"]
            elif header["name"] == "Permissions-Policy":
                print(header["value"])
                perm_policy = header["value"]

        return Response(status=status,
                        referrer_policy=ref_policy,
                        permission_policy=perm_policy,
                        accept_ch=accept_ch,
                        set_cookies=cookie_objects,
                        redirect_url=get_sld(urlparse(redirect).netloc) if len(
                            redirect) > 0 else ''
                        )

    @ staticmethod
    def _parse_entry(home_site: str, edir: EntryDir) -> Entry:
        return Entry(home_site=home_site,
                     request=EntryFactory._parse_request(home_site, edir),
                     response=EntryFactory._parse_response(edir))

    @staticmethod
    def entries_from_har(har_path: Union[str, Path]) -> Iterator[Entry]:
        har_path = Path(har_path)
        home_site = har_path.parts[-1].replace(".har", "")
        sld = get_sld(home_site)
        '''
        Use this method to parse a har file
        into entries of requests-responses.
        '''
        f = open(har_path, 'r')
        s = json_load(f)
        f.close()
        return [EntryFactory._parse_entry(sld, e) for e in s["log"]["entries"]]
